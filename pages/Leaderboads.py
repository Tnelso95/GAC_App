import streamlit as st
import pandas as pd
import re
import io
import numpy as np

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

# Data cleaning functions for hitting and pitching data
def clean_hitting_data(hitting_df):
    hitting_df = hitting_df.dropna(subset=['ExitVelocity', 'LaunchAngle', 'ExitDirection'])
    hitting_df['Date'] = pd.to_datetime(hitting_df['Date'])
    hitting_df['ExitVelocity'] = pd.to_numeric(hitting_df['ExitVelocity'], errors='coerce')
    hitting_df['LaunchAngle'] = pd.to_numeric(hitting_df['LaunchAngle'], errors='coerce')
    hitting_df['ExitDirection'] = pd.to_numeric(hitting_df['ExitDirection'], errors='coerce')
    hitting_df['Distance'] = pd.to_numeric(hitting_df['Distance'], errors='coerce')
    return hitting_df

def clean_pitching_data(pitching_df):
    pitching_df = pitching_df.dropna(subset=['Pitch Type', 'Is Strike', 'Velocity'])
    pitching_df['Date'] = pd.to_datetime(pitching_df['Date'])
    pitching_df['Velocity'] = pd.to_numeric(pitching_df['Velocity'], errors='coerce')
    return pitching_df

# Function to generate leaderboard based on selected metric
def generate_leaderboard(df, report_type):
    if report_type == "Hitting":
        leaderboard = df[['Player Name', 'ExitVelocity', 'Distance']].dropna(subset=['ExitVelocity'])
        balls_in_play = leaderboard[leaderboard['ExitVelocity'] > 0]
        leaderboard = balls_in_play.groupby('Player Name').agg(
            max_exit_velocity=('ExitVelocity', 'max'),
            hard_hit_percentage=('ExitVelocity', lambda x: round((x >= 90).mean() * 100, 2)),
            max_distance=('Distance', 'max')
        ).reset_index()
        leaderboard = leaderboard.sort_values(by='max_exit_velocity', ascending=False)

    elif report_type == "Pitching":
        overall_leaderboard = df[['Player Name', 'Velocity', 'Is Strike']].dropna(subset=['Velocity'])
        overall_leaderboard = overall_leaderboard.groupby('Player Name').agg(
            max_velocity=('Velocity', 'max'),
            overall_strike_percentage=('Is Strike', lambda x: round((x == 'YES').mean() * 100, 2))
        ).reset_index()

        pitch_type_leaderboard = df[['Player Name', 'Pitch Type', 'Velocity', 'Is Strike']].dropna(subset=['Velocity'])
        pitch_type_leaderboard = pitch_type_leaderboard.groupby(['Player Name', 'Pitch Type']).agg(
            max_velocity=('Velocity', 'max'),
            avg_velocity=('Velocity', lambda x: round(x.mean(), 2)),
            strike_percentage=('Is Strike', lambda x: round((x == 'YES').mean() * 100, 2))
        ).reset_index()

        overall_leaderboard = overall_leaderboard.sort_values(by='max_velocity', ascending=False)
        pitch_type_leaderboard = pitch_type_leaderboard.sort_values(by='max_velocity', ascending=False)

        return overall_leaderboard, pitch_type_leaderboard

    return leaderboard

# Streamlit app layout
st.set_page_config(
    page_title="LeaderBoards",
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

st.title("LeaderBoards 📊")

# Read and clean the pitching and hitting datasets
pitching_df = clean_file('/Users/tomasnelson/Desktop/GAC_App/pages/CSV/pitching.csv')
hitting_df = clean_file('/Users/tomasnelson/Desktop/GAC_App/pages/CSV/hitting.csv')

if hitting_df is not None and pitching_df is not None:
    hitting_df = clean_hitting_data(hitting_df)
    pitching_df = clean_pitching_data(pitching_df)

    # Sidebar for leaderboard
    st.sidebar.subheader('Leaderboard')
    leaderboard_type = st.sidebar.selectbox('Select Leaderboard Type', ['Hitting', 'Pitching'])
    
    if leaderboard_type == 'Hitting':
        leaderboard = generate_leaderboard(hitting_df, leaderboard_type)
        st.markdown(f"### Leaderboard for {leaderboard_type} Players")
        st.dataframe(leaderboard.style.format({
            'max_exit_velocity': '{:.2f}',
            'hard_hit_percentage': '{:.2f}%',
            'max_distance': '{:.2f}'
        }).hide(axis='index'))
    else:
        overall_leaderboard, pitch_type_leaderboard = generate_leaderboard(pitching_df, leaderboard_type)
        st.markdown("### Overall Leaderboard for Pitching Players")
        st.dataframe(overall_leaderboard.style.format({
            'max_velocity': '{:.2f}',
            'overall_strike_percentage': '{:.2f}%'
        }).hide(axis='index'))
        st.markdown("### Pitch Type Leaderboard for Pitching Players")
        st.dataframe(pitch_type_leaderboard.style.format({
            'max_velocity': '{:.2f}',
            'avg_velocity': '{:.2f}',
            'strike_percentage': '{:.2f}%'
        }).hide(axis='index'))

else:
    st.error("Error cleaning the files. Please check the CSV paths.")
