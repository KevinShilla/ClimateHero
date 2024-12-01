import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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

@st.cache_resource
def predict_co2_emission(data, country):
    country_data = data[data['Country'] == country]

    # Ensure there is no missing data
    country_data = country_data.dropna(subset=['Energy_consumption', 'CO2_emission'])

    # Prepare input features (Energy consumption) and target variable (CO2 emission)
    X = country_data[['Energy_consumption']].values
    y = country_data['CO2_emission'].values

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)

    # Create future predictions for 2020 to 2030
    energy_forecast = predict_energy_consumption(data, country)
    future_energy = energy_forecast['Energy_consumption'].values.reshape(-1, 1)

    # Predict CO2 emissions for future years
    predicted_co2 = rf_model.predict(future_energy)

    # Prepare forecast data
    co2_forecast = pd.DataFrame({'Year': energy_forecast['Year'], 'CO2_emission': predicted_co2})

    return co2_forecast

st.sidebar.header("Filters")
selected_country = st.sidebar.selectbox("Select Country:", sorted(data['Country'].unique()))
selected_metric = st.sidebar.radio(
    "Select Metric to Display:",
    ("Normalized Comparison (Energy & CO2)", "Energy Consumption", "CO2 Emission")
)

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

# Add prediction section
st.header("Energy and CO2 Emission Forecasts")
st.write(f"Forecast for {selected_country} from 2020 to 2030:")

# Predict energy consumption and CO2 emissions
energy_forecast = predict_energy_consumption(data, selected_country)
co2_forecast = predict_co2_emission(data, selected_country)

# Combine predictions into one DataFrame
combined_forecast = pd.merge(energy_forecast, co2_forecast, on='Year')

if selected_metric == "Normalized Comparison (Energy & CO2)":
    # Normalize values for better comparison
    combined_forecast['Energy_norm'] = combined_forecast['Energy_consumption'] / combined_forecast['Energy_consumption'].max()
    combined_forecast['CO2_norm'] = combined_forecast['CO2_emission'] / combined_forecast['CO2_emission'].max()

    # Plot the normalized graph
    st.subheader(f"Normalized Energy Consumption and CO2 Emissions for {selected_country} (2020-2030)")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot normalized Energy Consumption
    ax.plot(
        combined_forecast['Year'],
        combined_forecast['Energy_norm'],
        label='Energy Consumption (Normalized)',
        color='blue',
        linewidth=2,
    )

    # Plot normalized CO2 Emissions
    ax.plot(
        combined_forecast['Year'],
        combined_forecast['CO2_norm'],
        label='CO2 Emissions (Normalized)',
        color='red',
        linewidth=2,
    )

    # Add titles, labels, and legend
    ax.set_title(f"Energy Consumption and CO2 Emissions for {selected_country} (2020-2030)", fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Normalized Values", fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True)

    # Render the plot in Streamlit
    st.pyplot(fig)

elif selected_metric == "Energy Consumption":
    st.subheader(f"Energy Consumption Prediction for {selected_country} (2020-2030)")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        energy_forecast['Year'],
        energy_forecast['Energy_consumption'],
        label='Energy Consumption',
        color='blue',
        linewidth=2,
    )
    ax.set_title(f"Energy Consumption Prediction for {selected_country} (2020-2030)", fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Energy Consumption", fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True)
    st.pyplot(fig)

elif selected_metric == "CO2 Emission":
    st.subheader(f"CO2 Emission Prediction for {selected_country} (2020-2030)")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        co2_forecast['Year'],
        co2_forecast['CO2_emission'],
        label='CO2 Emissions',
        color='red',
        linewidth=2,
    )
    ax.set_title(f"CO2 Emission Prediction for {selected_country} (2020-2030)", fontsize=14)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("CO2 Emission", fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True)
    st.pyplot(fig)
