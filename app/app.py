import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Set page layout to wide
st.set_page_config(layout="wide")

# Inject CSS to style buttons as green
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

# Load the data
@st.cache
def load_data():
    data = pd.read_csv('../data/energy.csv')
    # Ensure correct data types
    data['Year'] = data['Year'].astype(int)
    data['Energy Consumption'] = pd.to_numeric(data['Energy Consumption'], errors='coerce')
    data['CO2 Emission'] = pd.to_numeric(data['CO2 Emission'], errors='coerce')
    return data

data = load_data()

# Create two columns with ratios 1:2 (left:right)
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Options")

    # Available years in the data (1980-2019)
    available_years = sorted(data['Year'].unique())
    year = st.selectbox("Select Year:", available_years)

    # List of countries available in the data
    country_list = sorted(data['Country'].unique())
    country = st.selectbox("Select Country:", country_list)

    # Two green buttons for energy consumption or CO2 emission
    st.write("")  # Add some spacing
    energy_button = st.button("Energy Consumption")
    co2_button = st.button("CO2 Emission")

with col2:
    st.header("Graph")

    if year > 2019:
        st.write("Data is not available for years after 2019.")
    else:
        selected_data = data[(data['Country'] == country) & (data['Year'] == year)]

        if selected_data.empty:
            st.write(f"No data available for {country} in {year}.")
        else:
            if energy_button:
                value = selected_data['Energy Consumption'].values[0]
                st.write(f"Energy Consumption for {country} in {year}: {value}")
                fig, ax = plt.subplots()
                ax.bar([country], [value], color='green')
                ax.set_ylabel('Energy Consumption')
                st.pyplot(fig)
            elif co2_button:
                # Plot CO2 Emission
                value = selected_data['CO2 Emission'].values[0]
                st.write(f"CO2 Emission for {country} in {year}: {value}")
                fig, ax = plt.subplots()
                ax.bar([country], [value], color='green')
                ax.set_ylabel('CO2 Emission')
                st.pyplot(fig)
