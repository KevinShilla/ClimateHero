import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the data
data_path = '../data/energy.csv'  # Update with the correct path
df_energy = pd.read_csv(data_path)

# Streamlit app
st.title("Energy Consumption Forecast")
st.sidebar.header("User Input")

# Country selection
countries = df_energy['Country'].unique()
selected_country = st.sidebar.selectbox('Select a country', countries)

# Buttons for functionality
if st.sidebar.button("Predict Energy Consumption"):
    # Filter data for the selected country
    country_data = df_energy[df_energy['Country'] == selected_country]

    # Prepare data for Prophet
    country_data['ds'] = pd.to_datetime(country_data['Year'], format='%Y')
    country_data['y'] = country_data['Energy_consumption']

    # Create and fit the Prophet model
    model = Prophet(yearly_seasonality=True)
    model.fit(country_data)

    # Make future dataframe - Full Forecast
    future = model.make_future_dataframe(periods=10, freq='Y')

    # Predict future values
    forecast = model.predict(future)

    # Plot the forecast for the selected country
    st.write(f"Energy Consumption Forecast for {selected_country} (2019-2030)")
    fig, ax = plt.subplots()
    ax.plot(forecast['ds'], forecast['yhat'], color='blue')
    ax.set_title(f"Energy Consumption Forecast for {selected_country} (2019-2030)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Energy Consumption")
    ax.grid(True)
    st.pyplot(fig)

# Additional placeholder for CO2 Emissions (not implemented yet)
if st.sidebar.button("Predict CO2 Emissions"):
    st.write("CO2 Emissions prediction is not implemented yet.")
