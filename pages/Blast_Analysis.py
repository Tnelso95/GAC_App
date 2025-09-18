import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import plotly.express as px
import requests
import io

# Set page config
st.set_page_config(
    page_title='Swing Analysis',
    page_icon="⚾",
    layout="wide"
)

# Function to calculate Swing Length
def calc_swing_length(time_to_contact, bat_speed):
    return (time_to_contact / 1.3636) * bat_speed

# Function to calculate Swing Acceleration
def calc_swing_acceleration(bat_speed, swing_length):
    return 0.03343 * (bat_speed ** 2 / swing_length)

# Function to calculate Swing Score
def calc_swing_score(swing_acceleration, min_swing_acc=15, max_swing_acc=30):
    score = 20 + ((swing_acceleration - min_swing_acc) / (max_swing_acc - min_swing_acc)) * 60
    return max(min(score, 80), 20)  # Ensure score is within the 20-80 scale

@st.cache_data
def fetch_baseball_savant_data():
    """Fetch data from Baseball Savant and return as a pandas DataFrame."""
    url = "https://baseballsavant.mlb.com/leaderboard/custom"
    
    # Define the parameters for the request
    params = {
        "year": "2023",
        "type": "batter",
        "min": "25",  # Qualified batters
        "selections": "player_name,avg_swing_speed,attack_angle,avg_swing_length,vertical_swing_path",
        "sort": "xwoba",
        "sortDir": "desc",
        "csv": "true"
    }
    
    # Set headers to mimic a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/csv"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def find_similar_players(df, bat_speed, attack_angle, swing_length, vertical_bat_angle, n=3):
    """Find the n most similar players based on swing metrics."""
    # Normalize the input metrics
    metrics = np.array([[bat_speed, attack_angle, swing_length, vertical_bat_angle]])
    player_metrics = df[['avg_swing_speed', 'attack_angle', 'avg_swing_length', 'vertical_swing_path']].values
    
    # Calculate distances
    distances = cdist(metrics, player_metrics)
    
    # Get indices of n most similar players
    similar_indices = np.argsort(distances[0])[:n]
    
    return df.iloc[similar_indices]

# Title and description
st.title('⚾ Swing Analysis')
st.markdown("""
    Enter your swing metrics to get:
    1. A swing score (20-80 scale) based on your swing acceleration
    2. Three MLB players with similar swing characteristics
""")

# Input metrics
col1, col2 = st.columns(2)

with col1:
    st.subheader("Enter Your Swing Metrics")
    bat_speed = st.number_input('Bat Speed (mph)', min_value=50.0, max_value=90.0, value=65.0, step=0.1)
    attack_angle = st.number_input('Attack Angle (degrees)', min_value=-10.0, max_value=30.0, value=8.0, step=0.1)
    time_to_contact = st.number_input('Time to Contact (seconds)', min_value=0.1, max_value=0.2, value=0.15, step=0.001)
    vertical_bat_angle = st.number_input('Vertical Bat Angle (degrees)', min_value=-30.0, max_value=50.0, value=0.0, step=0.1)

# Calculate metrics
swing_length = calc_swing_length(time_to_contact, bat_speed)
swing_acceleration = calc_swing_acceleration(bat_speed, swing_length)
swing_score = calc_swing_score(swing_acceleration)

with col2:
    st.subheader("Your Swing Metrics")
    
    # Create metrics display
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        st.metric("Swing Length", f"{swing_length:.2f} ft")
        st.metric("Swing Score", f"{swing_score:.1f}")
    
    with metrics_col2:
        st.metric("Swing Acceleration", f"{swing_acceleration:.2f} g")
        
    # Create a gauge chart for the swing score
    fig = px.pie(
        values=[swing_score - 20, 80 - swing_score],
        names=['Score', 'Remaining'],
        hole=0.7,
        color_discrete_sequence=['#00B4D8', '#E0E0E0']
    )
    
    fig.update_layout(
        annotations=[dict(text=f"{swing_score:.1f}", x=0.5, y=0.5, font_size=40, showarrow=False)],
        showlegend=False,
        width=300,
        height=300,
        margin=dict(t=0, b=0, l=0, r=0)
    )
    
    st.plotly_chart(fig)

# Fetch MLB data and find similar players
st.subheader("Similar MLB Players")
df = fetch_baseball_savant_data()

if df is not None:
    similar_players = find_similar_players(
        df, 
        bat_speed, 
        attack_angle, 
        swing_length, 
        vertical_bat_angle
    )
    
    # Display similar players
    for idx, player in similar_players.iterrows():
        with st.expander(f"🏃 {player['player_name']}", expanded=True):
            cols = st.columns(4)
            cols[0].metric("Bat Speed", f"{player['avg_swing_speed']:.1f} mph")
            cols[1].metric("Attack Angle", f"{player['attack_angle']:.1f}°")
            cols[2].metric("Swing Length", f"{player['avg_swing_length']:.2f} ft")
            cols[3].metric("Vertical Bat Angle", f"{player['vertical_swing_path']:.1f}°")
            
            # Calculate similarity score (0-100)
            similarity = 100 * (1 - np.sqrt(
                ((bat_speed - player['avg_swing_speed'])/20)**2 +
                ((attack_angle - player['attack_angle'])/15)**2 +
                ((swing_length - player['avg_swing_length'])/2)**2 +
                ((vertical_bat_angle - player['vertical_swing_path'])/15)**2
            ) / 2)
            
            st.progress(similarity/100, text=f"Similarity Score: {similarity:.1f}%")
else:
    st.error("Unable to fetch MLB player data. Please try again later.")

