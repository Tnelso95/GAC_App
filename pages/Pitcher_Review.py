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

# Data cleaning functions for hitting and pitching data
def clean_pitching_data(pitching_df):
    required_columns = ['Pitch Type', 'Is Strike', 'Velocity', 'Date', 'Strike Zone Side', 'Strike Zone Height']
    optional_columns = ['HB (trajectory)', 'VB (trajectory)']
    for col in required_columns:
        if col not in pitching_df.columns:
            st.error(f"Missing required column in pitching data: {col}")
            return pd.DataFrame()
    for col in optional_columns:
        if col not in pitching_df.columns:
            pitching_df[col] = np.nan
    pitching_df = pitching_df.dropna(subset=['Pitch Type', 'Is Strike', 'Velocity'])
    pitching_df['Date'] = pd.to_datetime(pitching_df['Date'])
    pitching_df['Velocity'] = pd.to_numeric(pitching_df['Velocity'], errors='coerce')
    pitching_df['Strike Zone Side'] = pd.to_numeric(pitching_df['Strike Zone Side'], errors='coerce')
    pitching_df['Strike Zone Height'] = pd.to_numeric(pitching_df['Strike Zone Height'], errors='coerce')
    pitching_df['HB (trajectory)'] = pd.to_numeric(pitching_df['HB (trajectory)'], errors='coerce')
    pitching_df['VB (trajectory)'] = pd.to_numeric(pitching_df['VB (trajectory)'], errors='coerce')
    return pitching_df

# Function to generate pitcher player report with additional metrics
def pitcher_player_report(pitching_df, player_name):
    player_data = pitching_df[pitching_df['Player Name'].str.contains(player_name, case=False, na=False)]
    total_pitches = len(player_data)

    pitch_type_stats = player_data.groupby('Pitch Type').agg(
        avg_velocity=('Velocity', 'mean'),
        max_velocity=('Velocity', 'max'),
        avg_horizontal_movement=('HB (trajectory)', 'mean'),
        avg_vertical_movement=('VB (trajectory)', 'mean'),
        strike_percentage=('Is Strike', lambda x: (x == 'YES').mean() * 100),
        pitch_count=('Pitch Type', 'size')
    ).reset_index()

    pitch_type_stats['Usage'] = (pitch_type_stats['pitch_count'] / total_pitches) * 100
    pitch_type_stats = pitch_type_stats.sort_values(by='Usage', ascending=False)
    overall_strike_percentage = (player_data['Is Strike'] == 'YES').mean() * 100

    report = {
        'Player Name': player_name,
        'Total Pitches': total_pitches,
        'Overall Strike Percentage': overall_strike_percentage
    }

    for _, row in pitch_type_stats.iterrows():
        report[row['Pitch Type']] = {
            'Average Velocity': row['avg_velocity'],
            'Max Velocity': row['max_velocity'],
            'Average Horizontal Movement': row['avg_horizontal_movement'],
            'Average Vertical Movement': row['avg_vertical_movement'],
            'Strike Percentage': row['strike_percentage'],
            'Usage': row['Usage']
        }

    return report

# Function to format the report for better readability and aesthetics
def format_report(report):
    formatted_report = f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.6;">
        <h2>Player Report</h2>
        <p><strong>Player Name:</strong> {report['Player Name']}</p>
        <p><strong>Total Pitches:</strong> {report['Total Pitches']}</p>
        <p><strong>Overall Strike Percentage:</strong> {report['Overall Strike Percentage']:.2f}%</p>
        <hr>
    """
    
    for pitch_type, stats in report.items():
        if pitch_type not in ['Player Name', 'Total Pitches', 'Overall Strike Percentage']:
            formatted_report += f"<h3>{pitch_type}</h3>"
            formatted_report += f"<p><strong>Average Velocity:</strong> {stats['Average Velocity']:.2f}</p>"
            formatted_report += f"<p><strong>Max Velocity:</strong> {stats['Max Velocity']:.2f}</p>"
            if not pd.isna(stats['Average Horizontal Movement']):
                formatted_report += f"<p><strong>Average Horizontal Movement:</strong> {stats['Average Horizontal Movement']:.2f}</p>"
            if not pd.isna(stats['Average Vertical Movement']):
                formatted_report += f"<p><strong>Average Vertical Movement:</strong> {stats['Average Vertical Movement']:.2f}</p>"
            formatted_report += f"<p><strong>Strike Percentage:</strong> {stats['Strike Percentage']:.2f}%</p>"
            formatted_report += f"<p><strong>Usage:</strong> {stats['Usage']:.2f}%</p>"
            formatted_report += "<hr>"
    
    formatted_report += "</div>"
    return formatted_report

# Function to plot jointplot for a specific player and pitch type
def plot_jointplot(player_data, player_name, pitch_type):
    joint_plot = sns.jointplot(x='Strike Zone Side', y='Strike Zone Height', data=player_data, kind='scatter', height=6, ratio=5, space=0.1)
    joint_plot.plot_joint(sns.kdeplot, levels=5)
    
    # Add strike zone box
    strike_zone = plt.Rectangle((-10, 15), 20, 25, fill=False, edgecolor='black', linewidth=2)
    joint_plot.ax_joint.add_patch(strike_zone)
    
    # Set consistent plot limits to zoom out further
    joint_plot.ax_joint.set_xlim(-30, 30)
    joint_plot.ax_joint.set_ylim(-20, 60)
    
    plt.suptitle(f'Jointplot of {pitch_type} Outcomes for {player_name}', y=1.02)
    st.pyplot(joint_plot)

# Streamlit app layout
st.set_page_config(
    page_title='Pitcher Review',
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

st.title('Pitcher Analysis and Review')


# Read and clean the pitching and hitting datasets
pitching_df = clean_file('/Users/tomasnelson/Desktop/GAC_App/pages/CSV/pitching.csv')

if pitching_df is not None:
    pitching_df = clean_pitching_data(pitching_df)

    if not pitching_df.empty:
        # Get unique player names for dropdowns
        all_players = sorted(set(pitching_df['Player Name'].dropna().unique()))

        # Sidebar for player report
        st.sidebar.subheader('Player Report')
        player_name = st.sidebar.selectbox('Select Player Name for Report', all_players)
        
        if player_name:
            report = pitcher_player_report(pitching_df, player_name)
            formatted_report = format_report(report)
            st.write(f"Report for Player: {player_name}")
            st.markdown(formatted_report, unsafe_allow_html=True)
            
            pitch_types = report.keys()
            for pitch_type in pitch_types:
                if pitch_type not in ['Player Name', 'Total Pitches', 'Overall Strike Percentage']:
                    player_data = pitching_df[(pitching_df['Player Name'] == player_name) & (pitching_df['Pitch Type'] == pitch_type)]
                    plot_jointplot(player_data, player_name, pitch_type)
    else:
        st.error("Error: Cleaned data is empty. Please check the CSV files.")
else:
    st.error("Error cleaning the files. Please check the CSV paths.")
