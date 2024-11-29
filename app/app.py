import streamlit as st

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

# Create two columns with ratios 1:2 (left:right)
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Options")

    # Let the user pick a year from 2020 to 2030
    year = st.selectbox("Select Year:", list(range(2020, 2031)))

    # Let the user pick a country
    country = st.text_input("Enter Country:")

    # Two green buttons for energy consumption or CO2 emission
    st.write("")  # Add some spacing
    energy_button = st.button("Energy Consumption")
    co2_button = st.button("CO2 Emission")

with col2:
    st.header("Graph")

    # Display an empty graph placeholder based on button clicks
    if energy_button:
        st.write(f"Energy Consumption Graph for {country} in {year}")
        st.empty()  # Placeholder for the graph
    elif co2_button:
        st.write(f"CO2 Emission Graph for {country} in {year}")
        st.empty()  # Placeholder for the graph
