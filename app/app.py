import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os
import pickle
import altair as alt



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

with open("data/country_flag_emoji.pkl", "rb") as f:
    country_flag_emoji = pickle.load(f)

def filter_renewables(data, country):
    renewables = data[(data['Country'] == country) & (data['Energy_type'] == 'renewables_n_other')]
    return renewables[['Year', 'Energy_consumption', 'Energy_production']]



#working:
def calculate_global_rankings(data):
    # Prepare forecasted data
    forecasted_data = pd.DataFrame()

    # Loop through each country
    for country in data['Country'].unique():
        renewable_forecast = forecast_renewables(data, country)
        energy_forecast = predict_energy_consumption(data, country)
        co2_forecast = predict_co2_emission(data, country)

        if renewable_forecast.empty or energy_forecast.empty or co2_forecast.empty:
            continue

        # Combine forecasts into a single DataFrame
        combined_forecast = renewable_forecast.merge(energy_forecast, on='Year').merge(co2_forecast, on='Year')
        combined_forecast['Country'] = country
        forecasted_data = pd.concat([forecasted_data, combined_forecast], ignore_index=True)

    # Calculate rankings
    forecasted_data['Renewable_rank'] = forecasted_data.groupby('Year')['Forecasted_consumption'].rank(ascending=False)
    forecasted_data['CO2_rank'] = forecasted_data.groupby('Year')['CO2_emission'].rank(ascending=True)
    forecasted_data['Consumption_rank'] = forecasted_data.groupby('Year')['Energy_consumption'].rank(ascending=True)

    # Calculate Climate Hero Points
    forecasted_data['Climate_points'] = (
        forecasted_data['Renewable_rank'] +
        forecasted_data['CO2_rank'] +
        forecasted_data['Consumption_rank']
    )

    # Average points by country
    rankings = forecasted_data.groupby('Country', as_index=False).agg({
        'Climate_points': 'mean',
        'Renewable_rank': 'mean',
        'CO2_rank': 'mean',
        'Consumption_rank': 'mean',  # Ensuring this column is included
    })
    rankings = rankings.sort_values(by='Climate_points', ascending=True)

    return rankings


def is_climate_hero(rankings, selected_country):
    # Get the row for the selected country
    row = rankings[rankings['Country'] == selected_country]

    if row.empty:
        return False

    # Check if the ranks are in the top 5
    return (
        row.iloc[0]['Consumption_rank'] <= 5 and
        row.iloc[0]['Renewable_rank'] <= 5
    )

   
   

def top_climate_heroes(rankings, top_n=5):
    return rankings.head(top_n)

def is_climate_hero(rankings, country):
    top_heroes = top_climate_heroes(rankings)
    return country in top_heroes['Country'].values



@st.cache_resource
def forecast_renewables(data, country):
    renewables = filter_renewables(data, country)
    if renewables.empty:
        return pd.DataFrame()

    # Prepare data for Prophet
    prophet_data = pd.DataFrame({
        'ds': pd.to_datetime(renewables['Year'], format='%Y'),
        'y': renewables['Energy_consumption']
    })
    
    model = Prophet()
    model.fit(prophet_data)
    future = model.make_future_dataframe(periods=11, freq='Y')
    forecast = model.predict(future)

    forecast['Year'] = forecast['ds'].dt.year
    forecast = forecast[['Year', 'yhat']].rename(columns={'yhat': 'Forecasted_consumption'})
    return forecast

#population is in million person in the data
@st.cache_resource
def predict_population(data, country):
    country_data = data[data['Country'] == country]
    prophet_data = pd.DataFrame({
        'ds': pd.to_datetime(country_data['Year'], format='%Y'),
        'y': country_data['Population']
    })

    model = Prophet()
    model.fit(prophet_data)

    future = model.make_future_dataframe(periods=12, freq='Y')
    forecast = model.predict(future)

    forecast = forecast[(forecast['ds'].dt.year >= 2020) & (forecast['ds'].dt.year <= 2030)]
    forecast['Year'] = forecast['ds'].dt.year
    forecast['Population'] = forecast['yhat']

    return forecast[['Year', 'Population']]

#Energy_consumption - Amount of Consumption for the specific energy source, measured (quad Btu)
@st.cache_resource
def predict_energy_consumption(data, country):
    country_data = data[data['Country'] == country]
    prophet_data = pd.DataFrame({
        'ds': pd.to_datetime(country_data['Year'], format='%Y'),
        'y': country_data['Energy_consumption']
    })

    model = Prophet()
    model.fit(prophet_data)

    future = model.make_future_dataframe(periods=12, freq='Y')
    forecast = model.predict(future)

    forecast = forecast[(forecast['ds'].dt.year >= 2020) & (forecast['ds'].dt.year <= 2030)]
    forecast['Year'] = forecast['ds'].dt.year
    forecast['Energy_consumption'] = forecast['yhat']

    return forecast[['Year', 'Energy_consumption']]

#CO2_emission - The amount of C02 emitted, measured (MMtonnes CO2) million metric tonnes 
@st.cache_resource
def predict_co2_emission(data, country):
    country_data = data[data['Country'] == country]

    # I did this just to make sure there is no missing data
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
# Get the flag for the selected country
flag = country_flag_emoji.get(selected_country, None)
if flag:
    # Display the flag with increased size
    st.sidebar.markdown(
        f"<span style='font-size: 250px;'>{flag}</span>",
        unsafe_allow_html=True
    )

selected_metric = st.sidebar.radio(
    "Select Metric to Display:",
    ("Normalized Comparison (Energy & CO2)", "Energy Consumption", "CO2 Emission")
)

st.title("Climate Hero")
st.write("Kevin Shilla & Sabeha Khan")
st.write("Our project analyzes historical energy consumption data and predicts future energy usage trends for different countries. It provides users with interactive visualizations, including energy consumption by type and forecasts up to 2030, helping to better understand global energy trends and make informed decisions.")

country_data = data[(data['Country'] == selected_country) & (data['Year'] <= 2019)]

# if not country_data.empty:
#     st.subheader(f"Historical Data for {selected_country} (1980-2019)")
#     st.dataframe(country_data)

#     fig, ax = plt.subplots(figsize=(12, 6))
#     energy_types = country_data['Energy_type'].unique()
#     colors = plt.cm.get_cmap('tab10', len(energy_types)).colors

#     for idx, energy_type in enumerate(energy_types):
#         type_data = country_data[country_data['Energy_type'] == energy_type]
#         ax.scatter(type_data['Year'], type_data['Energy_consumption'], label=energy_type, color=colors[idx])

#     ax.set_title(f"Energy Consumption by Type for {selected_country} (1980-2019)", fontsize=14)
#     ax.set_xlabel("Year", fontsize=12)
#     ax.set_ylabel("Energy Consumption", fontsize=12)
#     ax.legend(title="Energy Types", bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the graph
#     ax.grid(True, linestyle='--', linewidth=0.5)

#     st.pyplot(fig)

# else:
#     st.write(f"No historical data available for {selected_country} from 1980 to 2019.")

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

    # Ensure 'Year' is numeric and properly formatted
    combined_forecast['Year'] = combined_forecast['Year'].round().astype(int)

    # Prepare the data in long format for Altair
    chart_data = combined_forecast.melt(
        id_vars="Year",
        value_vars=["Energy_norm", "CO2_norm"],
        var_name="Metric",
        value_name="Normalized Value"
    )

    # Map metrics to custom names
    metric_mapping = {
       "Energy_norm": "Energy Consumption",
        "CO2_norm": "CO2 Emissions"
    }
    chart_data["Metric"] = chart_data["Metric"].map(metric_mapping)

    # Define custom color scale
    custom_colors = alt.Scale(
        domain=["CO2 Emissions", "Energy Consumption"],
        range=["red", "blue"]  # Red for CO2, Blue for Energy
    )

    # Get the min and max values from the data to avoid empty ranges
    y_min = chart_data['Normalized Value'].min()
    y_max = chart_data['Normalized Value'].max()

    # Create the Altair chart
    chart = (
        alt.Chart(chart_data)
        .mark_line(point=False)  # Add points for clarity
        .encode(
            x=alt.X("Year:O", title="Year", axis=alt.Axis(labelAngle=0)),  # Horizontal year labels
            y=alt.Y("Normalized Value:Q", title="Normalized Value", 
                    scale=alt.Scale(domain=[y_min, y_max]),),  # Show float values with 2 decimals
            color=alt.Color("Metric:N", title="Metric", scale=custom_colors),  # Custom colors
            tooltip=["Year", "Metric", "Normalized Value"]
        )
        .properties(width=800, height=400)
        .configure_view(stroke=None)  # Remove borders to avoid extra space
    ).interactive()

    # Display the chart in Streamlit
    st.subheader(f"Normalized Energy Consumption and CO2 Emissions for {selected_country} (2020-2030)")
    st.altair_chart(chart, use_container_width=True)

    # # Normalize values for better comparison
    # combined_forecast['Energy_norm'] = combined_forecast['Energy_consumption'] / combined_forecast['Energy_consumption'].max()
    # combined_forecast['CO2_norm'] = combined_forecast['CO2_emission'] / combined_forecast['CO2_emission'].max()

    # # Plot the normalized graph
    # st.subheader(f"Normalized Energy Consumption and CO2 Emissions for {selected_country} (2020-2030)")
    # fig, ax = plt.subplots(figsize=(12, 6))

    # # Plot normalized Energy Consumption
    # ax.plot(
    #     combined_forecast['Year'],
    #     combined_forecast['Energy_norm'],
    #     label='Energy Consumption (Normalized)',
    #     color='blue',
    #     linewidth=2,
    # )

    # # Plot normalized CO2 Emissions
    # ax.plot(
    #     combined_forecast['Year'],
    #     combined_forecast['CO2_norm'],
    #     label='CO2 Emissions (Normalized)',
    #     color='red',
    #     linewidth=2,
    # )

    # # Add titles, labels, and legend
    # ax.set_title(f"Energy Consumption and CO2 Emissions for {selected_country} (2020-2030)", fontsize=14)
    # ax.set_xlabel("Year", fontsize=12)
    # ax.set_ylabel("Normalized Values", fontsize=12)
    # ax.legend(fontsize=10, loc='upper left')
    # ax.grid(True)

    # # Render the plot in Streamlit
    # st.pyplot(fig)

elif selected_metric == "Energy Consumption":
    # Ensure 'Year' is numeric and properly formatted
    energy_forecast['Year'] = energy_forecast['Year'].round().astype(int)

    # Calculate dynamic y-axis domain based on actual energy consumption values
    y_min = energy_forecast['Energy_consumption'].min()
    y_max = energy_forecast['Energy_consumption'].max()

    # Create the Altair chart
    chart = (
        alt.Chart(energy_forecast)
        .mark_line(color='blue', size=2)  # Line chart with blue color and thicker line
        .encode(
            x=alt.X('Year:O', title='Year', axis=alt.Axis(labelAngle=0)),  # Horizontal labels for Year
            y=alt.Y(
                'Energy_consumption:Q',
                title='Energy Consumption',
                scale=alt.Scale(domain=[y_min, y_max]),  # Dynamic y-axis domain
                  # Format y-axis with 2 decimal points
            ),
            tooltip=['Year', 'Energy_consumption']  # Enable tooltips for better interaction
        )
        .properties(
            title=f"Energy Consumption Prediction for {selected_country} (2020-2030)",  # Chart title
            width=800,  # Chart width
            height=400  # Chart height
        )
        .configure_axis(
            grid=True,  # Add gridlines for better readability
            labelAngle=0  # Horizontal labels for Year
        )
        .configure_title(fontSize=14)  # Adjust title font size
    ).interactive()

    # Display the Altair chart in Streamlit
    st.subheader(f"Energy Consumption Prediction for {selected_country} (2020-2030)")
    st.altair_chart(chart, use_container_width=True)

    # st.subheader(f"Energy Consumption Prediction for {selected_country} (2020-2030)")
    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.plot(
    #     energy_forecast['Year'],
    #     energy_forecast['Energy_consumption'],
    #     label='Energy Consumption',
    #     color='blue',
    #     linewidth=2,
    # )
    # ax.set_title(f"Energy Consumption Prediction for {selected_country} (2020-2030)", fontsize=14)
    # ax.set_xlabel("Year", fontsize=12)
    # ax.set_ylabel("Energy Consumption", fontsize=12)
    # ax.legend(fontsize=10, loc='upper left')
    # ax.grid(True)
    # st.pyplot(fig)

elif selected_metric == "CO2 Emission":
    # Ensure 'Year' is numeric and properly formatted
    co2_forecast['Year'] = co2_forecast['Year'].round().astype(int)

    # Calculate dynamic y-axis domain based on actual CO2 emission values
    y_min = co2_forecast['CO2_emission'].min()
    y_max = co2_forecast['CO2_emission'].max()

    # Create the Altair chart
    chart = (
        alt.Chart(co2_forecast)
        .mark_line(color='red', size=2)  # Line chart with red color and thicker line
        .encode(
            x=alt.X('Year:O', title='Year', axis=alt.Axis(labelAngle=0)),  # Horizontal labels for Year
            y=alt.Y(
                'CO2_emission:Q',
                title='CO2 Emission',
                scale=alt.Scale(domain=[y_min, y_max]),  # Dynamic y-axis domain
            ),
            tooltip=['Year', 'CO2_emission']  # Enable tooltips for better interaction
        )
        .properties(
            title=f"CO2 Emission Prediction for {selected_country} (2020-2030)",  # Chart title
            width=800,  # Chart width
            height=400  # Chart height
        )
        .configure_axis(
            grid=True,  # Add gridlines for better readability
            labelAngle=0  # Horizontal labels for Year
        )
        .configure_title(fontSize=14)  # Adjust title font size
    ).interactive()

    # Display the Altair chart in Streamlit
    st.subheader(f"CO2 Emission Prediction for {selected_country} (2020-2030)")
    st.altair_chart(chart, use_container_width=True)

    # st.subheader(f"CO2 Emission Prediction for {selected_country} (2020-2030)")
    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.plot(
    #     co2_forecast['Year'],
    #     co2_forecast['CO2_emission'],
    #     label='CO2 Emissions',
    #     color='red',
    #     linewidth=2,
    # )
    # ax.set_title(f"CO2 Emission Prediction for {selected_country} (2020-2030)", fontsize=14)
    # ax.set_xlabel("Year", fontsize=12)
    # ax.set_ylabel("CO2 Emission", fontsize=12)
    # ax.legend(fontsize=10, loc='upper left')
    # ax.grid(True)
    # st.pyplot(fig)

renewable_data = filter_renewables(data, selected_country)

rankings_file = 'results/rankings.csv'

# If the file exists, load it
if os.path.exists(rankings_file):
    rankings = pd.read_csv(rankings_file)
else:
    # Otherwise, calculate the rankings and save them to a file
    rankings = calculate_global_rankings(data)
    rankings.to_csv(rankings_file, index=False)


df_sorted = rankings.sort_values(by='Climate_points', ascending=True)

# Add a Climate_rank column based on the sorted order
df_sorted['Climate_rank'] = range(1, len(df_sorted) + 1)

# Reset index if needed
df_sorted.reset_index(drop=True, inplace=True)
is_hero = is_climate_hero(df_sorted, selected_country)
st.subheader("Climate Hero Rankings")
if is_hero:
        st.success(f"{selected_country} is on track to become a Climate Hero by 2030!")
        st.balloons()
else:
    st.warning(f"{selected_country} is not a Climate Hero by 2030.")

st.table(df_sorted.head())


if not country_data.empty:
    
    # Create a mapping dictionary for legend labels
    energy_type_mapping = {
        "all_energy_types": "All Energy Types",
        "coal": "Coal",
        "natural_gas": "Natural Gas",
        "petroleum_n_other_liquids": "Petroleum & Other Liquids",
        "renewables_n_other": "Renewables & Other"
    }
    # Ensure 'Year' is numeric and properly formatted
    country_data['Year'] = country_data['Year'].round().astype(int)
     # Map the 'Energy_type' column to the new labels
    country_data['Energy_type'] = country_data['Energy_type'].map(energy_type_mapping)

    # Create the Altair chart
    chart = (
        alt.Chart(country_data)
        .mark_circle(size=100, opacity=0.7)  # Use circles for scatter plot
        .encode(
            x=alt.X('Year:O', title='Year', axis=alt.Axis(labelAngle=0)),  # Horizontal year labels
            y=alt.Y('Energy_consumption:Q', title='Energy Consumption'),
            color=alt.Color('Energy_type:N', title='Energy Types'),  # Color by Energy Type
            tooltip=['Year', 'Energy_consumption', 'Energy_type']  # Add tooltips
        )
        .properties(
            title=f"Energy Consumption by Type for {selected_country} (1980-2019)",
            width=800,
            height=400
        )
        .configure_axis(grid=True)  # Enable gridlines
        .configure_legend(
            titleFontSize=12,
            labelFontSize=10,
            orient='right'  # Place legend to the right
        )
        .configure_title(fontSize=14)  # Set title font size
    ).interactive()

    # Display the Altair chart in Streamlit
    st.altair_chart(chart, use_container_width=True)

    # fig, ax = plt.subplots(figsize=(12, 6))
    # energy_types = country_data['Energy_type'].unique()
    # colors = plt.cm.get_cmap('tab10', len(energy_types)).colors

    # for idx, energy_type in enumerate(energy_types):
    #     type_data = country_data[country_data['Energy_type'] == energy_type]
    #     ax.scatter(type_data['Year'], type_data['Energy_consumption'], label=energy_type, color=colors[idx])

    # ax.set_title(f"Energy Consumption by Type for {selected_country} (1980-2019)", fontsize=14)
    # ax.set_xlabel("Year", fontsize=12)
    # ax.set_ylabel("Energy Consumption", fontsize=12)
    # ax.legend(title="Energy Types", bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside the graph
    # ax.grid(True, linestyle='--', linewidth=0.5)

    # st.pyplot(fig)

    st.subheader(f"Historical Data for {selected_country} (1980-2019)")
    st.dataframe(country_data)

else:
    st.write(f"No historical data available for {selected_country} from 1980 to 2019.")
