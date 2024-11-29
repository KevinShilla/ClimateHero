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

    # Let the user pick a year from 1980 to 2030
    year = st.selectbox("Select Year:", list(range(1980, 2031)))

    # List of countries to select from
    country_list = [
        'Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Argentina',
        'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain',
        'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin',
        'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil',
        'Brunei', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon',
        'Canada', 'Cape Verde', 'Central African Republic', 'Chad', 'Chile',
        'China', 'Colombia', 'Comoros', 'Costa Rica', 'Croatia', 'Cuba',
        'Cyprus', 'Czech Republic', 'Denmark', 'Djibouti', 'Dominica',
        'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador',
        'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia',
        'Fiji', 'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany',
        'Ghana', 'Greece', 'Grenada', 'Guatemala', 'Guinea', 'Guyana', 'Haiti',
        'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq',
        'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan',
        'Kazakhstan', 'Kenya', 'Kiribati', 'Kuwait', 'Kyrgyzstan', 'Laos',
        'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein',
        'Lithuania', 'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia',
        'Maldives', 'Mali', 'Malta', 'Mauritania', 'Mauritius', 'Mexico',
        'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique',
        'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Zealand',
        'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'North Macedonia',
        'Norway', 'Oman', 'Pakistan', 'Palau', 'Panama', 'Papua New Guinea',
        'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar',
        'Romania', 'Russia', 'Rwanda', 'Saint Kitts and Nevis', 'Saint Lucia',
        'Samoa', 'San Marino', 'Saudi Arabia', 'Senegal', 'Serbia',
        'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia',
        'Solomon Islands', 'Somalia', 'South Africa', 'South Korea', 'Spain',
        'Sri Lanka', 'Sudan', 'Suriname', 'Sweden', 'Switzerland', 'Syria',
        'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo',
        'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan',
        'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom',
        'United States', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Vatican City',
        'Venezuela', 'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe'
    ]

    # Let the user pick a country from the list
    country = st.selectbox("Select Country:", country_list)

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
