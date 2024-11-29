import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

st.set_page_config(layout="wide")

st.markdown("""
    <style>
    .stButton > button {
        background-color: green;
        color: white;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    data = pd.read_csv('data/energy.csv')
    data['Year'] = data['Year'].astype(int)
    data['Energy_consumption'] = pd.to_numeric(data['Energy_consumption'], errors='coerce')
    data['CO2_emission'] = pd.to_numeric(data['CO2_emission'], errors='coerce')
    return data

data = load_data()

@st.cache_resource
def predict_energy_consumption(data, country):
    country_data = data[data['Country'] == country]
    prophet_data = pd.DataFrame({
        'ds': pd.to_datetime(country_data['Year'], format='%Y'),
        'y': country_data['Energy_consumption']
    })

    model = Prophet()
    model.fit(prophet_data)

    future = model.make_future_dataframe(periods=11, freq='Y')
    forecast = model.predict(future)

    forecast = forecast[(forecast['ds'].dt.year >= 2020) & (forecast['ds'].dt.year <= 2030)]
    forecast['Year'] = forecast['ds'].dt.year
    forecast['Energy_consumption'] = forecast['yhat']

    return forecast[['Year', 'Energy_consumption']]

st.sidebar.header("Filters")
selected_year = st.sidebar.selectbox("Select Year:", sorted(data['Year'].unique()))
selected_country = st.sidebar.selectbox("Select Country:", sorted(data['Country'].unique()))
selected_metric = st.sidebar.radio(
    "Select Metric to Display:",
    ("Energy Consumption", "CO2 Emission")
)

st.sidebar.subheader("Forecast Options")
forecast_start_year = st.sidebar.slider(
    "Select Start Year for Forecast:", min_value=2019, max_value=2029, value=2019
)

filtered_data = data[(data['Year'] == selected_year) & (data['Country'] == selected_country)]

st.title("Climate Vision")
st.write("Kevin Shilla & Sabeha Khan")
st.write("Our project analyzes historical energy consumption data and predicts future energy usage trends for different countries. It provides users with interactive visualizations, including energy consumption by type and forecasts up to 2030, helping to better understand global energy trends and make informed decisions.")

country_data = data[(data['Country'] == selected_country) & (data['Year'] <= 2019)]

if not country_data.empty:
    st.subheader(f"Historical Data for {selected_country} (1980-2019)")
    st.dataframe(country_data)

    fig, ax = plt.subplots(figsize=(12, 6))
    energy_types = country_data['Energy_type'].unique()
    colors = plt.cm.get_cmap('tab10', len(energy_types)).colors

    for idx, energy_type in enumerate(energy_types):
        type_data = country_data[country_data['Energy_type'] == energy_type]
        ax.scatter(type_data['Year'], type_data['Energy_consumption'], label=energy_type, color=colors[idx])

    ax.set_title(f"Energy Consumption by Type for {selected_country} (1980-2019)", fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Energy Consumption", fontsize=12)
    ax.legend(title="Energy Types", bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the graph
    ax.grid(True, linestyle='--', linewidth=0.5)

    st.pyplot(fig)

else:
    st.write(f"No historical data available for {selected_country} from 1980 to 2019.")

st.header("Energy Consumption Prediction")
st.write(f"Forecast for {selected_country} from {forecast_start_year} to 2030:")

forecast_data = predict_energy_consumption(data, selected_country)

forecast_data = forecast_data[forecast_data['Year'] >= forecast_start_year]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(forecast_data['Year'], forecast_data['Energy_consumption'], color='blue', linewidth=2)
ax.set_title(f"Energy Consumption Forecast for {selected_country} ({forecast_start_year}-2030)", fontsize=14)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Energy Consumption", fontsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
st.pyplot(fig)
