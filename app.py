import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import json
import geopandas as gpd
from shapely.geometry import Point

# Load datasets
covid_state_data = pd.read_csv(r'covid cases by state_Full Data_data.csv')
covid_india = pd.read_csv(r'Data/covid_19_india.csv')
vaccine_data = pd.read_csv(r'Data/covid_vaccine_statewise.csv')
covid2_new = pd.read_csv(r'Covid2_new.csv')
testing_data = pd.read_csv(r'Data/StatewiseTestingDetails.csv')
merged_data = pd.read_csv(r'Data/merged_covid_vaccine_data.csv')

# Fix date columns
covid_state_data['Date'] = pd.to_datetime(covid_state_data['Date'])
covid_india['Date'] = pd.to_datetime(covid_india['Date'])
covid2_new['Date'] = pd.to_datetime(covid2_new['Date'])
testing_data['Date'] = pd.to_datetime(testing_data['Date'])
merged_data['Date'] = pd.to_datetime(merged_data['Date'])

# Filter for second wave (March 2021 to July 2021)
start_date = '2021-03-01'
end_date = '2021-07-31'

covid_state_wave2 = covid_state_data[(covid_state_data['Date'] >= start_date) & (covid_state_data['Date'] <= end_date)]
covid_india_wave2 = covid_india[(covid_india['Date'] >= start_date) & (covid_india['Date'] <= end_date)]
covid2_new_wave2 = covid2_new[(covid2_new['Date'] >= start_date) & (covid2_new['Date'] <= end_date)]
merged_wave2 = merged_data[(merged_data['Date'] >= start_date) & (merged_data['Date'] <= end_date)]

# State coordinates for mapping (approximate centroids)
state_coordinates = {
    'Maharashtra': [19.7515, 75.7139],
    'Kerala': [10.8505, 76.2711],
    'Karnataka': [15.3173, 75.7139],
    'Tamil Nadu': [11.1271, 78.6569],
    'Andhra Pradesh': [15.9129, 79.7400],
    'Uttar Pradesh': [26.8467, 80.9462],
    'Delhi': [28.7041, 77.1025],
    'West Bengal': [22.9868, 87.8550],
    'Rajasthan': [27.0238, 74.2179],
    'Gujarat': [22.2587, 71.1924],
    'Madhya Pradesh': [22.9734, 78.6569],
    'Haryana': [29.0588, 76.0856],
    'Bihar': [25.0961, 85.3131],
    'Telangana': [18.1124, 79.0193],
    'Odisha': [20.9517, 85.0985],
    'Punjab': [31.1471, 75.3412],
    'Assam': [26.2006, 92.9376],
    'Jharkhand': [23.6102, 85.2799],
    'Uttarakhand': [30.0668, 79.0193],
    'Chhattisgarh': [21.2787, 81.8661],
    'Himachal Pradesh': [31.1048, 77.1734],
    'Goa': [15.2993, 74.1240],
    'Jammu and Kashmir': [33.7782, 76.5762],
    'Puducherry': [11.9416, 79.8083],
    'Tripura': [23.9408, 91.9882],
    'Chandigarh': [30.7333, 76.7794],
    'Manipur': [24.6637, 93.9063],
    'Meghalaya': [25.4670, 91.3662],
    'Nagaland': [26.1584, 94.5624],
    'Arunachal Pradesh': [28.2180, 94.7278],
    'Mizoram': [23.1645, 92.9376],
    'Sikkim': [27.5330, 88.5122],
    'Dadra and Nagar Haveli': [20.1809, 73.0169],
    'Daman and Diu': [20.4283, 72.8397],
    'Andaman and Nicobar Islands': [11.7401, 92.6586],
    'Lakshadweep': [10.5667, 72.6417],
    'Ladakh': [34.1526, 77.5770]
}

# Function to prepare data for PyDeck
def prepare_data_for_pydeck(df, date_snapshot):
    df_snapshot = df[df['Date'] == date_snapshot].copy()
    
    # Create DataFrame with state coordinates
    data_with_coords = []
    
    for index, row in df_snapshot.iterrows():
        state = row['State']
        # Clean up state names to match our coordinates dictionary
        if state in state_coordinates:
            lat, lon = state_coordinates[state]
            new_row = row.to_dict()
            new_row['latitude'] = lat
            new_row['longitude'] = lon
            data_with_coords.append(new_row)
    
    return pd.DataFrame(data_with_coords)

# Create aggregated state data with peak metrics during second wave
peak_data = []

for state in covid2_new_wave2['State'].unique():
    state_data = covid2_new_wave2[covid2_new_wave2['State'] == state]
    
    # Find peak metrics
    peak_cases = state_data['Confirmed'].max()
    peak_new_cases = state_data['New Cases'].max()
    peak_deaths = state_data['Deaths'].max()
    peak_date = state_data.loc[state_data['New Cases'].idxmax(), 'Date']
    
    # Calc total metrics for the wave
    total_cases = state_data['New Cases'].sum()
    total_deaths = state_data['Deaths'].max() - state_data['Deaths'].min()
    
    # Get coordinates
    if state in state_coordinates:
        lat, lon = state_coordinates[state]
        
        peak_data.append({
            'State': state,
            'Peak_Daily_Cases': peak_new_cases,
            'Peak_Date': peak_date,
            'Total_Wave2_Cases': total_cases,
            'Total_Wave2_Deaths': total_deaths,
            'CFR': (total_deaths / total_cases * 100) if total_cases > 0 else 0,
            'latitude': lat,
            'longitude': lon
        })

peak_df = pd.DataFrame(peak_data)

# Create a column view for the hover tooltip
def create_description(row):
    return f"{row['State']}<br>Peak Daily Cases: {int(row['Peak_Daily_Cases']):,}<br>Total Wave 2 Cases: {int(row['Total_Wave2_Cases']):,}<br>CFR: {row['CFR']:.2f}%"

peak_df['description'] = peak_df.apply(create_description, axis=1)

# 1. PEAK COVID CASES MAP VISUALIZATION
def create_peak_cases_map(peak_df):
    # Normalize the radius values
    peak_df['radius'] = peak_df['Peak_Daily_Cases'] / peak_df['Peak_Daily_Cases'].max() * 100000
    
    # Create the layer
    layer = pdk.Layer(
        "ScatterplotLayer",
        peak_df,
        get_position=["longitude", "latitude"],
        get_radius="radius",
        get_fill_color=[255, 0, 0, 140],  # Red with alpha
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
    )
    
    # Set the view state
    view_state = pdk.ViewState(
        longitude=78.9629, 
        latitude=22.5937,  # Center of India
        zoom=4,
        min_zoom=3,
        max_zoom=10,
        pitch=0,
        bearing=0
    )
    
    # Create the tooltip
    tooltip = {
        "html": "<b>{State}</b><br>Peak Daily Cases: {Peak_Daily_Cases}<br>On: {Peak_Date}<br>Total Wave2 Cases: {Total_Wave2_Cases}<br>CFR: {CFR}%",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }
    
    # Create the deck
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/light-v9"
    )
    
    return deck

# 2. CHOROPLETH MAP FOR CFR (CASE FATALITY RATE)
# def create_cfr_map(peak_df):
#     # Define a color scale for CFR
#     peak_df['color_r'] = (peak_df['CFR'] / peak_df['CFR'].max() * 255).astype(int)
    
#     layer = pdk.Layer(
#         "ScatterplotLayer",
#         peak_df,
#         get_position=["longitude", "latitude"],
#         get_radius=50000,  # Fixed radius
#         get_fill_color=["color_r", 10, 10, 200],  # Red scale based on CFR
#         pickable=True,
#         opacity=0.8,
#         stroked=True,
#         filled=True,
#     )
    
#     view_state = pdk.ViewState(
#         longitude=78.9629, 
#         latitude=22.5937,  # Center of India
#         zoom=4,
#         min_zoom=3,
#         max_zoom=10,
#         pitch=0,
#         bearing=0
#     )
    
#     tooltip = {
#         "html": "<b>{State}</b><br>Case Fatality Rate: {CFR:.2f}%<br>Total Deaths: {Total_Wave2_Deaths}",
#         "style": {"backgroundColor": "darkred", "color": "white"}
#     }
    
#     deck = pdk.Deck(
#         layers=[layer],
#         initial_view_state=view_state,
#         tooltip=tooltip,
#         map_style="mapbox://styles/mapbox/light-v9"
#     )
    
#     return deck

# 3. TIME SERIES VISUALIZATION FOR TOP STATES
def prepare_time_series_data(covid2_new_wave2):
    # Get top 5 states by total cases
    top_states = covid2_new_wave2.groupby('State')['Confirmed'].max().nlargest(5).index.tolist()
    
    # Filter for these states
    top_states_data = covid2_new_wave2[covid2_new_wave2['State'].isin(top_states)]
    
    # Create a time series dataset
    time_series = []
    
    for state in top_states:
        state_data = top_states_data[top_states_data['State'] == state].sort_values('Date')
        
        for _, row in state_data.iterrows():
            time_series.append({
                'State': state,
                'Date': row['Date'],
                'New_Cases': row['New Cases'],
                'Confirmed': row['Confirmed'],
                'Deaths': row['Deaths'],
                'latitude': state_coordinates[state][0],
                'longitude': state_coordinates[state][1]
            })
    
    return pd.DataFrame(time_series)

# 4. VACCINATION ANALYSIS
def analyze_vaccination_data(merged_wave2):
    # Group by state and date, then calculate stats
    vax_data = merged_wave2.groupby(['State', 'Date']).agg({
        'Total Doses Administered': 'max',
        'First Dose Administered': 'max',
        'Second Dose Administered': 'max',
        'Male (Doses Administered)': 'max',
        'Female (Doses Administered)': 'max'
    }).reset_index()
    
    # Get the last record for each state to show final vaccination status
    final_vax = vax_data.sort_values('Date').groupby('State').last().reset_index()
    
    # Add coordinates
    final_vax_with_coords = []
    for index, row in final_vax.iterrows():
        state = row['State']
        if state in state_coordinates:
            lat, lon = state_coordinates[state]
            new_row = row.to_dict()
            new_row['latitude'] = lat
            new_row['longitude'] = lon
            final_vax_with_coords.append(new_row)
    
    return pd.DataFrame(final_vax_with_coords)

# Create the vaccination visualization
def create_vaccination_map(vax_data):
    # Normalize for visualization
    max_doses = vax_data['Total Doses Administered'].max()
    vax_data['radius'] = vax_data['Total Doses Administered'] / max_doses * 100000
    
    layer = pdk.Layer(
        "ScatterplotLayer",
        vax_data,
        get_position=["longitude", "latitude"],
        get_radius="radius",
        get_fill_color=[0, 128, 255, 140],  # Blue with alpha
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
    )
    
    view_state = pdk.ViewState(
        longitude=78.9629, 
        latitude=22.5937,  # Center of India
        zoom=4,
        min_zoom=3,
        max_zoom=10,
        pitch=0,
        bearing=0
    )
    
    tooltip = {
        "html": "<b>{State}</b><br>Total Doses: {Total Doses Administered:,.0f}<br>First Doses: {First Dose Administered:,.0f}<br>Second Doses: {Second Dose Administered:,.0f}",
        "style": {"backgroundColor": "royalblue", "color": "white"}
    }
    
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/light-v9"
    )
    
    return deck

# 5. VACCINATION vs CASE CORRELATION
def vaccination_vs_cases(merged_wave2):
    # Group by state to get final statistics
    state_stats = merged_wave2.groupby('State').agg({
        'Total Doses Administered': 'max',
        'Confirmed': 'max',
        'Deaths': 'max'
    }).reset_index()
    
    # Add coordinates
    state_stats_with_coords = []
    for index, row in state_stats.iterrows():
        state = row['State']
        if state in state_coordinates:
            lat, lon = state_coordinates[state]
            new_row = row.to_dict()
            new_row['latitude'] = lat
            new_row['longitude'] = lon
            
            # Calculate cases per 100k vaccines
            if row['Total Doses Administered'] > 0:
                new_row['Cases_per_100k_Vaccines'] = (row['Confirmed'] / row['Total Doses Administered']) * 100000
            else:
                new_row['Cases_per_100k_Vaccines'] = 0
                
            state_stats_with_coords.append(new_row)
    
    return pd.DataFrame(state_stats_with_coords)

# IMPLEMENTATION - COMBINE ALL VISUALIZATIONS IN A DASHBOARD

# Prepare all datasets
peak_cases_data = peak_df
# cfr_data = peak_df
time_series_data = prepare_time_series_data(covid2_new_wave2)
vaccination_data = analyze_vaccination_data(merged_wave2)
vax_vs_cases = vaccination_vs_cases(merged_wave2)

# CREATE THE MAIN FUNCTION TO GENERATE THE DASHBOARD
def generate_covid_dashboard():
    # Create all the visualization decks
    peak_cases_deck = create_peak_cases_map(peak_cases_data)
    # cfr_deck = create_cfr_map(cfr_data)
    vaccination_deck = create_vaccination_map(vaccination_data)
    
    # For demonstration in a notebook environment
    # Note: For a deployed application, you would use appropriate layout frameworks like Dash or Streamlit
    
    print("COVID-19 Second Wave Dashboard (March-July 2021)")
    print("="*50)
    print("\n1. PEAK COVID CASES MAP VISUALIZATION")
    peak_cases_deck.to_html("peak_cases_map.html")
    
    # print("\n2. CASE FATALITY RATE (CFR) MAP")
    # cfr_deck.to_html("cfr_map.html")
    
    print("\n2. VACCINATION PROGRESS MAP")
    vaccination_deck.to_html("vaccination_map.html")
    
    # Create additional analytics as HTML files for complete dashboard
    # This section would use matplotlib, seaborn or plotly to generate additional charts
    
    # Return the main deck for display
    return peak_cases_deck

# Call the function to generate the dashboard
covid_dashboard = generate_covid_dashboard()

# Function to animate the progression of cases over time
def create_time_animation(covid2_new_wave2):
    # Get unique dates to animate through
    dates = sorted(covid2_new_wave2['Date'].unique())
    
    # Sample every 7 days to make animation manageable
    sampled_dates = dates[::7]
    
    animation_frames = []
    
    for date in sampled_dates:
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        day_data = prepare_data_for_pydeck(covid2_new_wave2, date)
        
        if not day_data.empty:
            # Normalize radius based on confirmed cases
            max_cases = day_data['Confirmed'].max()
            if max_cases > 0:
                day_data['radius'] = day_data['Confirmed'] / max_cases * 100000
            else:
                day_data['radius'] = 10000  # Default radius if no cases
                
            layer = pdk.Layer(
                "ScatterplotLayer",
                day_data,
                get_position=["longitude", "latitude"],
                get_radius="radius",
                get_fill_color=[255, 0, 0, 140],
                pickable=True,
                opacity=0.8,
                stroked=True,
                filled=True,
            )
            
            view_state = pdk.ViewState(
                longitude=78.9629, 
                latitude=22.5937,
                zoom=4,
                min_zoom=3,
                max_zoom=10,
                pitch=0,
                bearing=0
            )
            
            tooltip = {
                "html": "<b>{State}</b><br>Date: {Date}<br>Confirmed: {Confirmed}<br>Deaths: {Deaths}",
                "style": {"backgroundColor": "steelblue", "color": "white"}
            }
            
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip,
                map_style="mapbox://styles/mapbox/light-v9"
            )
            
            # Save this frame
            html_path = f"animation_frame_{date_str}.html"
            deck.to_html(html_path)
            animation_frames.append(html_path)
            
            print(f"Generated frame for {date_str}")
    
    print(f"Created {len(animation_frames)} animation frames.")
    return animation_frames

# Additional function for detailed analysis
def generate_detailed_analysis():
    # 1. Top 5 states with highest cases
    top_case_states = covid2_new_wave2.groupby('State')['Confirmed'].max().nlargest(5)
    
    # 2. Top 5 states with highest deaths
    top_death_states = covid2_new_wave2.groupby('State')['Deaths'].max().nlargest(5)
    
    # 3. Calculate case fatality rates for all states
    cfr_by_state = []
    
    for state in covid2_new_wave2['State'].unique():
        state_data = covid2_new_wave2[covid2_new_wave2['State'] == state]
        first_record = state_data.iloc[0]
        last_record = state_data.iloc[-1]
        
        total_cases = last_record['Confirmed'] - first_record['Confirmed']
        total_deaths = last_record['Deaths'] - first_record['Deaths']
        
        cfr = (total_deaths / total_cases * 100) if total_cases > 0 else 0
        
        cfr_by_state.append({
            'State': state,
            'Total Cases': total_cases,
            'Total Deaths': total_deaths,
            'CFR (%)': cfr
        })
    
    cfr_df = pd.DataFrame(cfr_by_state)
    top_cfr_states = cfr_df.sort_values('CFR (%)', ascending=False).head(5)
    
    # 4. Vaccination progress correlation with cases (if data available)
    # This would be implemented if the merged_wave2 dataset has the necessary data
    
    # Print the analysis results
    print("\nDETAILED ANALYSIS OF COVID-19 SECOND WAVE (MARCH-JULY 2021)")
    print("="*70)
    
    print("\nTop 5 States by Confirmed Cases:")
    print(top_case_states)
    
    print("\nTop 5 States by Deaths:")
    print(top_death_states)
    
    print("\nTop 5 States by Case Fatality Rate:")
    print(top_cfr_states)
    
    return {
        'top_cases': top_case_states,
        'top_deaths': top_death_states,
        'top_cfr': top_cfr_states
    }

# Run the dashboard and analysis
covid_dashboard = generate_covid_dashboard()
detailed_analysis = generate_detailed_analysis()
animation_frames = create_time_animation(covid2_new_wave2)


import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Load the merged dataset that contains both COVID cases and vaccination data
merged_data = pd.read_csv(r'Data/merged_covid_vaccine_data.csv')

# Convert date column to datetime
merged_data['Date'] = pd.to_datetime(merged_data['Date'])

# Filter for second wave period (March 2021 to July 2021)
start_date = '2021-03-01'
end_date = '2021-07-31'
wave2_data = merged_data[(merged_data['Date'] >= start_date) & (merged_data['Date'] <= end_date)]

# Calculate rolling averages for smoother trends
def prepare_state_trends(state_data):
    # Sort by date
    state_data = state_data.sort_values('Date')
    
    # Calculate 7-day rolling averages
    state_data['Cases_7day_avg'] = state_data['New Cases'].rolling(7).mean()
    
    # Calculate daily vaccination rate (first difference of cumulative doses)
    state_data['Daily_Doses'] = state_data['Total Doses Administered'].diff()
    state_data['Doses_7day_avg'] = state_data['Daily_Doses'].rolling(7).mean()
    
    return state_data

# Function to create visualization for a specific state
def create_state_vaccination_impact(state_name):
    state_data = wave2_data[wave2_data['State'] == state_name].copy()
    
    if len(state_data) < 10:  # Skip if insufficient data
        return None
    
    state_data = prepare_state_trends(state_data)
    
    # Create a dual-axis plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add new cases trend
    fig.add_trace(
        go.Scatter(
            x=state_data['Date'], 
            y=state_data['Cases_7day_avg'],
            name="New Cases (7-day avg)",
            line=dict(color='red', width=2)
        ),
        secondary_y=False
    )
    
    # Add vaccination trend
    fig.add_trace(
        go.Scatter(
            x=state_data['Date'], 
            y=state_data['Doses_7day_avg'],
            name="Daily Vaccinations (7-day avg)",
            line=dict(color='blue', width=2)
        ),
        secondary_y=True
    )
    
    # Add titles and labels
    fig.update_layout(
        title_text=f"COVID-19 Cases vs. Vaccination Rate: {state_name}",
        xaxis_title="Date",
        legend=dict(y=0.99, x=0.01, orientation='h'),
        template='plotly_white',
        height=600
    )
    
    fig.update_yaxes(title_text="New Cases (7-day avg)", secondary_y=False)
    fig.update_yaxes(title_text="Daily Vaccinations (7-day avg)", secondary_y=True)
    
    return fig

# Function to analyze vaccination impact across all states
def analyze_vaccination_impact():
    # Get list of top 10 states by total cases
    case_totals = wave2_data.groupby('State')['Confirmed'].max().nlargest(10)
    top_states = case_totals.index.tolist()
    
    # Calculate metrics to measure vaccination impact
    impact_metrics = []
    
    for state in top_states:
        state_data = wave2_data[wave2_data['State'] == state].copy()
        
        if len(state_data) < 30:  # Skip if insufficient data
            continue
            
        state_data = state_data.sort_values('Date')
        
        # Get vaccination and case data
        first_record = state_data.iloc[0]
        peak_record = state_data.loc[state_data['New Cases'].idxmax()]
        last_record = state_data.iloc[-1]
        
        # Calculate metrics
        pre_vax_cases = first_record['Confirmed']
        peak_cases = peak_record['Confirmed']
        end_cases = last_record['Confirmed']
        
        start_vax = first_record['Total Doses Administered'] if not pd.isna(first_record['Total Doses Administered']) else 0
        peak_vax = peak_record['Total Doses Administered'] if not pd.isna(peak_record['Total Doses Administered']) else 0
        end_vax = last_record['Total Doses Administered'] if not pd.isna(last_record['Total Doses Administered']) else 0
        
        # Calculate the case decline rate after peak
        days_after_peak = (last_record['Date'] - peak_record['Date']).days
        if days_after_peak > 0:
            case_decline_rate = (peak_cases - end_cases) / days_after_peak
        else:
            case_decline_rate = 0
            
        # Calculate vaccination rate
        total_days = (last_record['Date'] - first_record['Date']).days
        if total_days > 0:
            vax_rate = (end_vax - start_vax) / total_days
        else:
            vax_rate = 0
        
        impact_metrics.append({
            'State': state,
            'Case_Growth_to_Peak': peak_cases - pre_vax_cases,
            'Case_Decline_After_Peak': end_cases - peak_cases,
            'Daily_Case_Decline_Rate': case_decline_rate,
            'Daily_Vaccination_Rate': vax_rate,
            'Total_Vaccinations': end_vax,
            'Peak_Date': peak_record['Date'],
            'Peak_Cases': peak_record['New Cases']
        })
    
    impact_df = pd.DataFrame(impact_metrics)
    
    # Create visualization of relationship between vaccination rate and case decline
    fig = px.scatter(
        impact_df, 
        x='Daily_Vaccination_Rate', 
        y='Daily_Case_Decline_Rate',
        size='Total_Vaccinations',
        color='Peak_Cases',
        hover_name='State',
        text='State',
        color_continuous_scale='Viridis',
        title='Relationship Between Vaccination Rate and COVID-19 Case Decline Rate',
        labels={
            'Daily_Vaccination_Rate': 'Daily Vaccination Rate (doses/day)',
            'Daily_Case_Decline_Rate': 'Daily Case Decline Rate (cases/day)',
            'Total_Vaccinations': 'Total Vaccinations',
            'Peak_Cases': 'Peak Daily Cases'
        },
        size_max=50
    )
    
    fig.update_traces(textposition='top center')
    fig.update_layout(height=700, template='plotly_white')
    
    return fig, impact_df

# Function to create vaccination distribution analysis
def analyze_vaccination_distribution():
    # Filter the last available record for each state to get final vaccination numbers
    latest_records


import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import json
import base64

# Load datasets
def load_datasets():
    covid_state_data = pd.read_csv(r'Data/covid cases by state_Full Data_data.csv')
    covid_india = pd.read_csv(r'Data/covid_19_india.csv')
    vaccine_data = pd.read_csv(r'Data/covid_vaccine_statewise.csv')
    covid2_new = pd.read_csv(r'Covid2_new.csv')
    testing_data = pd.read_csv(r'Data/StatewiseTestingDetails.csv')
    merged_data = pd.read_csv(r'Data/merged_covid_vaccine_data.csv')
    age_group_data = pd.read_csv(r'Data/AgeGroupDetails.csv')
    
    # Fix date columns
    covid_state_data['Date'] = pd.to_datetime(covid_state_data['Date'])
    covid_india['Date'] = pd.to_datetime(covid_india['Date'])
    covid2_new['Date'] = pd.to_datetime(covid2_new['Date'])
    testing_data['Date'] = pd.to_datetime(testing_data['Date'])
    merged_data['Date'] = pd.to_datetime(merged_data['Date'])
    
    return {
        'covid_state': covid_state_data,
        'covid_india': covid_india,
        'vaccine': vaccine_data,
        'covid2': covid2_new,
        'testing': testing_data,
        'merged': merged_data,
        'age_group': age_group_data,
    }

# Filter for second wave period (March 2021 to July 2021)
def filter_wave2_data(datasets):
    start_date = '2021-03-01'
    end_date = '2021-07-31'
    
    wave2_datasets = {}
    
    for key, df in datasets.items():
        if 'Date' in df.columns:
            wave2_datasets[key] = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        else:
            wave2_datasets[key] = df
    
    return wave2_datasets

# State coordinates for mapping (approximate centroids)
state_coordinates = {
    'Maharashtra': [19.7515, 75.7139],
    'Kerala': [10.8505, 76.2711],
    'Karnataka': [15.3173, 75.7139],
    'Tamil Nadu': [11.1271, 78.6569],
    'Andhra Pradesh': [15.9129, 79.7400],
    'Uttar Pradesh': [26.8467, 80.9462],
    'Delhi': [28.7041, 77.1025],
    'West Bengal': [22.9868, 87.8550],
    'Rajasthan': [27.0238, 74.2179],
    'Gujarat': [22.2587, 71.1924],
    'Madhya Pradesh': [22.9734, 78.6569],
    'Haryana': [29.0588, 76.0856],
    'Bihar': [25.0961, 85.3131],
    'Telangana': [18.1124, 79.0193],
    'Odisha': [20.9517, 85.0985],
    'Punjab': [31.1471, 75.3412],
    'Assam': [26.2006, 92.9376],
    'Jharkhand': [23.6102, 85.2799],
    'Uttarakhand': [30.0668, 79.0193],
    'Chhattisgarh': [21.2787, 81.8661],
    'Himachal Pradesh': [31.1048, 77.1734],
    'Goa': [15.2993, 74.1240],
    'Jammu and Kashmir': [33.7782, 76.5762],
    'Puducherry': [11.9416, 79.8083],
    'Tripura': [23.9408, 91.9882],
    'Chandigarh': [30.7333, 76.7794],
    'Manipur': [24.6637, 93.9063],
    'Meghalaya': [25.4670, 91.3662],
    'Nagaland': [26.1584, 94.5624],
    'Arunachal Pradesh': [28.2180, 94.7278],
    'Mizoram': [23.1645, 92.9376],
    'Sikkim': [27.5330, 88.5122],
    'Dadra and Nagar Haveli': [20.1809, 73.0169],
    'Daman and Diu': [20.4283, 72.8397],
    'Andaman and Nicobar Islands': [11.7401, 92.6586],
    'Lakshadweep': [10.5667, 72.6417],
    'Ladakh': [34.1526, 77.5770]
}

# Create peak cases data for map visualization
def create_peak_cases_data(covid2_wave2):
    peak_data = []
    
    for state in covid2_wave2['State'].unique():
        state_data = covid2_wave2[covid2_wave2['State'] == state]
        
        # Find peak metrics
        peak_cases = state_data['Confirmed'].max()
        peak_new_cases = state_data['New Cases'].max()
        peak_deaths = state_data['Deaths'].max()
        peak_date = state_data.loc[state_data['New Cases'].idxmax(), 'Date'] if not state_data['New Cases'].isna().all() else None
        
        # Calculate total metrics for the wave
        total_cases = state_data['New Cases'].sum()
        total_deaths = state_data['Deaths'].max() - state_data['Deaths'].min()
        
        # Get coordinates
        if state in state_coordinates:
            lat, lon = state_coordinates[state]
            
            peak_data.append({
                'State': state,
                'Peak_Daily_Cases': peak_new_cases,
                'Peak_Date': peak_date,
                'Total_Wave2_Cases': total_cases,
                'Total_Wave2_Deaths': total_deaths,
                'CFR': (total_deaths / total_cases * 100) if total_cases > 0 else 0,
                'latitude': lat,
                'longitude': lon
            })
    
    return pd.DataFrame(peak_data)

# Create PyDeck visualization for peak COVID cases
def create_peak_cases_map(peak_df):
    # Normalize the radius values
    peak_df['radius'] = peak_df['Peak_Daily_Cases'] / peak_df['Peak_Daily_Cases'].max() * 100000
    
    # Create the layer
    layer = pdk.Layer(
        "ScatterplotLayer",
        peak_df,
        get_position=["longitude", "latitude"],
        get_radius="radius",
        get_fill_color=[255, 0, 0, 140],  # Red with alpha
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
    )
    
    # Set the view state
    view_state = pdk.ViewState(
        longitude=78.9629, 
        latitude=22.5937,  # Center of India
        zoom=4,
        min_zoom=3,
        max_zoom=10,
        pitch=0,
        bearing=0
    )
    
    # Create the tooltip
    tooltip = {
        "html": "<b>{State}</b><br>Peak Daily Cases: {Peak_Daily_Cases}<br>On: {Peak_Date}<br>Total Wave2 Cases: {Total_Wave2_Cases}<br>CFR: {CFR}%",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }
    
    # Create the deck
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/light-v9"
    )
    
    return deck

# Create vaccination data for map visualization
def create_vaccination_data(merged_wave2):
    # Group by state and date, then calculate stats
    vax_data = merged_wave2.groupby(['State', 'Date']).agg({
        'Total Doses Administered': 'max',
        'First Dose Administered': 'max',
        'Second Dose Administered': 'max',
    }).reset_index()
    
    # Get the last record for each state to show final vaccination status
    final_vax = vax_data.sort_values('Date').groupby('State').last().reset_index()
    
    # Add coordinates
    final_vax_with_coords = []
    for index, row in final_vax.iterrows():
        state = row['State']
        if state in state_coordinates:
            lat, lon = state_coordinates[state]
            new_row = row.to_dict()
            new_row['latitude'] = lat
            new_row['longitude'] = lon
            final_vax_with_coords.append(new_row)
    
    return pd.DataFrame(final_vax_with_coords)

# Create PyDeck visualization for vaccination data
def create_vaccination_map(vax_data):
    # Normalize for visualization
    max_doses = vax_data['Total Doses Administered'].max()
    vax_data['radius'] = vax_data['Total Doses Administered'] / max_doses * 100000
    
    layer = pdk.Layer(
        "ScatterplotLayer",
        vax_data,
        get_position=["longitude", "latitude"],
        get_radius="radius",
        get_fill_color=[0, 128, 255, 140],  # Blue with alpha
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
    )
    
    view_state = pdk.ViewState(
        longitude=78.9629, 
        latitude=22.5937,  # Center of India
        zoom=4,
        min_zoom=3,
        max_zoom=10,
        pitch=0,
        bearing=0
    )
    
    tooltip = {
        "html": "<b>{State}</b><br>Total Doses: {Total Doses Administered:,.0f}<br>First Doses: {First Dose Administered:,.0f}<br>Second Doses: {Second Dose Administered:,.0f}",
        "style": {"backgroundColor": "royalblue", "color": "white"}
    }
    
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/light-v9"
    )
    
    return deck

# Create time series visualization for top states
def create_time_series_chart(covid2_wave2):
    # Get top 5 states by total cases
    top_states = covid2_wave2.groupby('State')['Confirmed'].max().nlargest(5).index.tolist()
    
    # Filter data for these states
    top_states_data = covid2_wave2[covid2_wave2['State'].isin(top_states)]
    
    # Create a time series chart
    fig = px.line(
        top_states_data, 
        x='Date', 
        y='New Cases',
        color='State',
        title='Daily New COVID-19 Cases in Top 5 States (Second Wave)',
        labels={'New Cases': 'Daily New Cases', 'Date': 'Date'}
    )
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='New Cases',
        legend_title='State',
        hovermode='x unified',
        height=500
    )
    
    return fig

# Create vaccination progress chart over time
def create_vaccination_progress_chart(merged_wave2):
    # Group by date and sum vaccination data across states
    vax_progress = merged_wave2.groupby('Date')[
        ['Total Doses Administered', 'First Dose Administered', 'Second Dose Administered']
    ].sum().reset_index()
    
    # Replace NaN with 0
    vax_progress = vax_progress.fillna(0)
    
    # Create the plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=vax_progress['Date'], 
        y=vax_progress['First Dose Administered'],
        mode='lines',
        name='First Dose',
        line=dict(width=2, color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=vax_progress['Date'], 
        y=vax_progress['Second Dose Administered'],
        mode='lines',
        name='Second Dose',
        line=dict(width=2, color='green')
    ))
    
    fig.add_trace(go.Scatter(
        x=vax_progress['Date'], 
        y=vax_progress['Total Doses Administered'],
        mode='lines',
        name='Total Doses',
        line=dict(width=3, color='red')
    ))
    
    fig.update_layout(
        title='Cumulative COVID-19 Vaccination Progress (Second Wave)',
        xaxis_title='Date',
        yaxis_title='Number of Doses',
        legend_title='Dose Type',
        hovermode='x unified',
        height=500
    )
    
    return fig

# Create testing vs positivity rate chart
def create_testing_positivity_chart(testing_data):
    # Calculate positivity rate
    testing_data['Positivity_Rate'] = (testing_data['Positive'] / testing_data['TotalSamples']) * 100
    
    # Group by state and calculate average positivity rate
    state_positivity = testing_data.groupby('State')['Positivity_Rate'].mean().reset_index()
    state_positivity = state_positivity.sort_values('Positivity_Rate', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        state_positivity.head(15),  # Top 15 states
        x='State',
        y='Positivity_Rate',
        title='Average COVID-19 Test Positivity Rate by State (Second Wave)',
        color='Positivity_Rate',
        color_continuous_scale='Reds',
        labels={'Positivity_Rate': 'Positivity Rate (%)'}
    )
    
    fig.update_layout(
        xaxis_title='State',
        yaxis_title='Positivity Rate (%)',
        xaxis={'categoryorder':'total descending'},
        height=500
    )
    
    return fig

# Create recovery rate vs vaccination rate scatter plot
def create_recovery_vaccination_scatter(merged_wave2):
    # Calculate recovery rate and vaccination rate by state
    state_metrics = []
    
    for state in merged_wave2['State'].unique():
        state_data = merged_wave2[merged_wave2['State'] == state].sort_values('Date')
        
        if len(state_data) < 2:
            continue
            
        # Calculate metrics
        first_record = state_data.iloc[0]
        last_record = state_data.iloc[-1]
        
        # Recovery rate
        total_confirmed = last_record['Confirmed'] - first_record['Confirmed']
        total_recovered = last_record['Recovered'] - first_record['Recovered']
        
        if total_confirmed > 0:
            recovery_rate = (total_recovered / total_confirmed) * 100
        else:
            recovery_rate = 0
            
        # Vaccination rate
        if 'Total Doses Administered' in state_data.columns and not pd.isna(last_record['Total Doses Administered']):
            vax_rate = last_record['Total Doses Administered']
        else:
            vax_rate = 0
            
        state_metrics.append({
            'State': state,
            'Recovery_Rate': recovery_rate,
            'Vaccination_Rate': vax_rate,
            'Total_Cases': total_confirmed
        })
    
    state_metrics_df = pd.DataFrame(state_metrics)
    
    # Create scatter plot
    fig = px.scatter(
        state_metrics_df,
        x='Vaccination_Rate',
        y='Recovery_Rate',
        size='Total_Cases',
        color='Total_Cases',
        hover_name='State',
        text='State',
        title='Relationship Between Vaccination Rate and Recovery Rate by State',
        labels={
            'Vaccination_Rate': 'Total Vaccine Doses',
            'Recovery_Rate': 'Recovery Rate (%)',
            'Total_Cases': 'Total Cases'
        },
        color_continuous_scale='Viridis'
    )
    
    fig.update_traces(textposition='top center')
    
    fig.update_layout(
        xaxis_title='Total Vaccine Doses',
        yaxis_title='Recovery Rate (%)',
        height=600
    )
    
    return fig

# Create age group distribution chart
def create_age_distribution_chart(age_group_data):
    # Sort age groups properly
    age_group_data['sort_order'] = age_group_data['AgeGroup'].apply(
        lambda x: int(x.split('-')[0]) if '-' in x else 0
    )
    
    sorted_data = age_group_data.sort_values('sort_order')
    
    # Create bar chart
    fig = px.bar(
        sorted_data,
        x='AgeGroup',
        y='TotalCases',
        title='COVID-19 Cases by Age Group (Second Wave)',
        color='TotalCases',
        labels={'TotalCases': 'Total Cases', 'AgeGroup': 'Age Group'},
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        xaxis_title='Age Group',
        yaxis_title='Total Cases',
        height=400
    )
    
    return fig

# Create detailed state metrics table
def create_state_metrics_table(covid2_wave2):
    # Calculate metrics by state
    state_metrics = []
    
    for state in covid2_wave2['State'].unique():
        state_data = covid2_wave2[covid2_wave2['State'] == state].sort_values('Date')
        
        if len(state_data) < 2:
            continue
            
        # Calculate metrics
        first_record = state_data.iloc[0]
        last_record = state_data.iloc[-1]
        peak_record_index = state_data['New Cases'].idxmax() if not state_data['New Cases'].isna().all() else 0
        
        if peak_record_index != 0:
            peak_record = state_data.loc[peak_record_index]
            peak_date = peak_record['Date']
            peak_cases = peak_record['New Cases']
        else:
            peak_date = None
            peak_cases = 0
        
        # Basic metrics
        total_cases = last_record['Confirmed'] - first_record['Confirmed']
        total_deaths = last_record['Deaths'] - first_record['Deaths']
        total_recovered = last_record['Recovered'] - first_record['Recovered'] if 'Recovered' in state_data.columns else 0
        
        # Calculate rates
        cfr = (total_deaths / total_cases * 100) if total_cases > 0 else 0
        recovery_rate = (total_recovered / total_cases * 100) if total_cases > 0 else 0
        
        state_metrics.append({
            'State': state,
            'Total_Cases': total_cases,
            'Total_Deaths': total_deaths,
            'Peak_Daily_Cases': peak_cases,
            'Peak_Date': peak_date,
            'CFR': cfr,
            'Recovery_Rate': recovery_rate
        })
    
    state_metrics_df = pd.DataFrame(state_metrics)
    
    # Create table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['State', 'Total Cases', 'Total Deaths', 'Peak Daily Cases', 'Peak Date', 'CFR (%)', 'Recovery Rate (%)'],
            fill_color='paleturquoise',
            align='left'
        ),
        cells=dict(
            values=[
                state_metrics_df['State'],
                state_metrics_df['Total_Cases'].apply(lambda x: f"{x:,.0f}"),
                state_metrics_df['Total_Deaths'].apply(lambda x: f"{x:,.0f}"),
                state_metrics_df['Peak_Daily_Cases'].apply(lambda x: f"{x:,.0f}"),
                state_metrics_df['Peak_Date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else 'N/A'),
                state_metrics_df['CFR'].apply(lambda x: f"{x:.2f}"),
                state_metrics_df['Recovery_Rate'].apply(lambda x: f"{x:.2f}")
            ],
            fill_color='lavender',
            align='left'
        )
    )])
    
    fig.update_layout(
        title='Detailed COVID-19 Metrics by State (Second Wave)',
        height=600
    )
    
    return fig

# Create a dashboard using Dash
def create_dashboard():
    # Load and prepare data
    datasets = load_datasets()
    wave2_datasets = filter_wave2_data(datasets)
    
    # Create datasets for visualizations
    peak_cases_df = create_peak_cases_data(wave2_datasets['covid2'])
    vaccination_df = create_vaccination_data(wave2_datasets['merged'])
    
    # Create PyDeck visualizations
    peak_cases_map = create_peak_cases_map(peak_cases_df)
    vaccination_map = create_vaccination_map(vaccination_df)
    
    # Save PyDeck visualizations to HTML files
    peak_cases_map.to_html("peak_cases_map.html")
    vaccination_map.to_html("vaccination_map.html")
    
    # Create Plotly visualizations
    time_series_chart = create_time_series_chart(wave2_datasets['covid2'])
    vaccination_progress_chart = create_vaccination_progress_chart(wave2_datasets['merged'])
    testing_positivity_chart = create_testing_positivity_chart(wave2_datasets['testing'])
    recovery_vaccination_scatter = create_recovery_vaccination_scatter(wave2_datasets['merged'])
    age_distribution_chart = create_age_distribution_chart(wave2_datasets['age_group'])
    state_metrics_table = create_state_metrics_table(wave2_datasets['covid2'])
    
    # Initialize Dash app
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # App layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("COVID-19 Second Wave in India (March-July 2021)",
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginTop': 20})
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("COVID-19 Peak Daily Cases by State"),
                    dbc.CardBody([
                        html.Iframe(
                            id='peak-cases-map',
                            srcDoc=open('peak_cases_map.html', 'r').read(),
                            style={'width': '100%', 'height': '400px'}
                        )
                    ])
                ])
            ], width=12)
        ], className='mb-4'),
               
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Daily COVID-19 Cases in Top 5 States"),
                    dbc.CardBody([
                        dcc.Graph(figure=time_series_chart)
                    ])
                ])
            ], width=12)
        ], className='mb-4'),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Vaccination Progress Across India"),
                    dbc.CardBody([
                        html.Iframe(
                            id='vax-map',
                            srcDoc=open('vaccination_map.html', 'r').read(),
                            style={'width': '100%', 'height': '400px'}
                        )
                    ])
                ])
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Vaccination Progress Over Time"),
                    dbc.CardBody([
                        dcc.Graph(figure=vaccination_progress_chart)
                    ])
                ])
            ], width=6)
        ], className='mb-4'),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Test Positivity Rate by State"),
                    dbc.CardBody([
                        dcc.Graph(figure=testing_positivity_chart)
                    ])
                ])
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Vaccination vs Recovery Rate"),
                    dbc.CardBody([
                        dcc.Graph(figure=recovery_vaccination_scatter)
                    ])
                ])
            ], width=6)
        ], className='mb-4'),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Cases by Age Group"),
                    dbc.CardBody([
                        dcc.Graph(figure=age_distribution_chart)
                    ])
                ])
            ], width=12)
        ], className='mb-4'),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Detailed State Metrics"),
                    dbc.CardBody([
                        dcc.Graph(figure=state_metrics_table)
                    ])
                ])
            ], width=12)
        ], className='mb-4')
    ], fluid=True)
    
    return app

# Main function to run the dashboard
if __name__ == '__main__':
    app = create_dashboard()
    app.run(debug=True)
    
    print("Dashboard is running. Open http://127.0.0.1:8050/ in your web browser to view it.")
