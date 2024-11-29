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
@st.cache_data
def load_data():
    # Adjust the file path as needed
    data = pd.read_csv('../data/energy.csv')
    # Ensure correct data types
    data['Year'] = data['Year'].astype(int)
    data['Energy Consumption'] = pd.to_numeric(data['Energy Consumption'], errors='coerce')
    data['CO2 Emission'] = pd.to_numeric(data['CO2 Emission'], errors='coerce')
    return data

data = load_data()

# Rest of your code...
