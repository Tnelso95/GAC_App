import streamlit as st
import pandas as pd
import re
import io
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Function to clean files and handle non-ASCII characters
def clean_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            raw_data = file.read()
        clean_data = re.sub(r'[^\x00-\x7F]+', '', raw_data)
        return pd.read_csv(io.StringIO(clean_data))
    except Exception as e:
        st.error(f"Error cleaning the file: {e}")
        return None

# Data cleaning function for hitting data
def clean_hitting_data(hitting_df):
    required_columns = ['ExitVelocity', 'LaunchAngle', 'ExitDirection', 'Date', 'Distance']
    for col in required_columns:
        if col not in hitting_df.columns:
            st.error(f"Missing required column in hitting data: {col}")
            return pd.DataFrame()
    hitting_df = hitting_df.dropna(subset=['ExitVelocity', 'LaunchAngle', 'ExitDirection'])
    hitting_df['Date'] = pd.to_datetime(hitting_df['Date'])
    hitting_df['ExitVelocity'] = pd.to_numeric(hitting_df['ExitVelocity'], errors='coerce')
    hitting_df['LaunchAngle'] = pd.to_numeric(hitting_df['LaunchAngle'], errors='coerce')
    hitting_df['ExitDirection'] = pd.to_numeric(hitting_df['ExitDirection'], errors='coerce')
    hitting_df['Distance'] = pd.to_numeric(hitting_df['Distance'], errors='coerce')
    return hitting_df

# Function to generate hitter player report
def hitter_player_report(hitting_df, player_name):
    player_data = hitting_df[hitting_df['Player Name'].str.contains(player_name, case=False, na=False)]
    hits = player_data[player_data['ExitVelocity'] > 0]

    average_exit_velocity = hits['ExitVelocity'].mean() if not hits.empty else 0
    max_exit_velocity = hits['ExitVelocity'].max() if not hits.empty else 0
    max_distance = hits['Distance'].max() if not hits.empty else 0

    hard_hit_balls = hits[hits['ExitVelocity'] >= 90]
    hard_hit_percentage = (len(hard_hit_balls) / len(hits) * 100) if len(hits) > 0 else 0

    report = {
        'Player Name': player_name,
        'Average Exit Velocity': average_exit_velocity,
        'Max Exit Velocity': max_exit_velocity,
        'Max Distance': max_distance,
        'Hard Hit Balls Percentage': hard_hit_percentage
    }

    return report

# Function to format the report for better readability and aesthetics
def format_report(report):
    formatted_report = f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.6;">
        <h2>Player Report</h2>
        <p><strong>Player Name:</strong> {report['Player Name']}</p>
        <p><strong>Average Exit Velocity:</strong> {report['Average Exit Velocity']:.2f}</p>
        <p><strong>Max Exit Velocity:</strong> {report['Max Exit Velocity']:.2f}</p>
        <p><strong>Max Distance:</strong> {report['Max Distance']:.2f}</p>
        <p><strong>Hard Hit Balls Percentage:</strong> {report['Hard Hit Balls Percentage']:.2f}%</p>
    </div>
    """
    return formatted_report

# Streamlit app layout
st.set_page_config(
    page_title='Hitting Review ',
    page_icon="https://static.wixstatic.com/media/7383ad_9fe67936506547e19ad5985ef14c8142~mv2.png/v1/fill/w_400,h_400,al_c,q_85,usm_1.20_1.00_0.01,enc_avif,quality_auto/CBH%20Logo.png",
)

# Add custom CSS for background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://img.freepik.com/free-photo/light-background_24972-1415.jpg");
        background-size: cover; /* Adjust background size */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Hitting Data Analysis and Review')


# Read and clean the hitting dataset
hitting_df = clean_file('/Users/tomasnelson/Desktop/GAC_App/pages/CSV/hitting.csv')

if hitting_df is not None:
    hitting_df = clean_hitting_data(hitting_df)

    if not hitting_df.empty:
        # Get unique player names for dropdowns
        all_players = sorted(hitting_df['Player Name'].dropna().unique())

        # Sidebar for player report
        st.sidebar.subheader('Player Report')
        player_name = st.sidebar.selectbox('Select Player Name for Report', all_players)
        
        if player_name:
            report = hitter_player_report(hitting_df, player_name)
            formatted_report = format_report(report)
            st.write(f"Report for Player: {player_name}")
            st.markdown(formatted_report, unsafe_allow_html=True)
    else:
        st.error("Error: Cleaned data is empty. Please check the CSV file.")
else:
    st.error("Error cleaning the file. Please check the CSV path.")
