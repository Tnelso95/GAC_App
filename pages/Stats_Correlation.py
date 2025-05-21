import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from scipy import stats
import numpy as np
import io

# Set page config
st.set_page_config(
    page_title="MLB Stats Correlation Analyzer",
    page_icon="⚾",
    layout="wide"
)

# Define stat labels mapping
STAT_LABELS = {
    'player_age': 'Age',
    'ab': 'At Bats',
    'pa': 'Plate Appearances',
    'hit': 'Hits',
    'single': 'Singles',
    'double': 'Doubles',
    'triple': 'Triples',
    'home_run': 'Home Runs',
    'strikeout': 'Strikeouts',
    'walk': 'Walks',
    'k_percent': 'Strikeout %',
    'bb_percent': 'Walk %',
    'batting_avg': 'Batting Average',
    'slg_percent': 'Slugging %',
    'on_base_percent': 'On-Base %',
    'on_base_plus_slg': 'OPS',
    'isolated_power': 'Isolated Power',
    'babip': 'BABIP',
    'xba': 'Expected Batting Avg',
    'xslg': 'Expected Slugging',
    'woba': 'wOBA',
    'xwoba': 'Expected wOBA',
    'xobp': 'Expected On-Base %',
    'xiso': 'Expected Isolated Power',
    'wobacon': 'wOBA on Contact',
    'xwobacon': 'Expected wOBA on Contact',
    'bacon': 'Batting Avg on Contact',
    'xbacon': 'Expected Batting Avg on Contact',
    'xbadiff': 'xBA - BA Difference',
    'xslgdiff': 'xSLG - SLG Difference',
    'wobadiff': 'xwOBA - wOBA Difference',
    'avg_swing_speed': 'Bat Speed (mph)',
    'fast_swing_rate': 'Fast Swing Rate',
    'blasts_contact': 'Blast Contact Rate',
    'blasts_swing': 'Blast Swing Rate',
    'squared_up_contact': 'Squared Up Contact Rate',
    'squared_up_swing': 'Squared Up Swing Rate',
    'avg_swing_length': 'Avg Swing Length',
    'swords': 'Swords Rate',
    'attack_angle': 'Attack Angle',
    'attack_direction': 'Attack Direction',
    'ideal_angle_rate': 'Ideal Angle Rate',
    'vertical_swing_path': 'Vertical Swing Path',
    'exit_velocity_avg': 'Avg Exit Velocity (mph)',
    'launch_angle_avg': 'Avg Launch Angle',
    'sweet_spot_percent': 'Sweet Spot %',
    'barrel_batted_rate': 'Barrel Rate',
    'solidcontact_percent': 'Solid Contact %',
    'flareburner_percent': 'Flare/Burner %',
    'poorlyunder_percent': 'Under %',
    'poorlytopped_percent': 'Topped %',
    'poorlyweak_percent': 'Weak %',
    'hard_hit_percent': 'Hard Hit %',
    'avg_best_speed': 'EV50',
    'avg_hyper_speed': 'Adjusted EV',
    'z_swing_percent': 'Zone Swing %',
    'z_swing_miss_percent': 'Zone Swing & Miss %',
    'oz_swing_percent': 'Out of Zone Swing %',
    'oz_swing_miss_percent': 'Out of Zone Swing & Miss %',
    'oz_contact_percent': 'Out of Zone Contact %',
    'meatball_swing_percent': 'Meatball Swing %',
    'iz_contact_percent': 'In Zone Contact %',
    'whiff_percent': 'Whiff %',
    'swing_percent': 'Swing %',
    'pull_percent': 'Pull %',
    'straightaway_percent': 'Straightaway %',
    'opposite_percent': 'Oppo%',
    'groundballs_percent': 'Ground Ball %',
    'flyballs_percent': 'Fly Ball %',
    'linedrives_percent': 'Line Drive %',
    'popups_percent': 'Pop Up %'
}

# Add custom CSS
st.markdown("""
    <style>
    .stPlot {
        background-color: #ffffff;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMarkdown {
        padding: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("⚾ MLB Stats Correlation Analyzer")
st.markdown("""
    Analyze correlations between different MLB statistics from Baseball Savant.
    Select any two metrics to visualize their relationship and see detailed statistical analysis.
""")

@st.cache_data
def fetch_baseball_savant_data():
    """Fetch data from Baseball Savant and return as a pandas DataFrame."""
    url = "https://baseballsavant.mlb.com/leaderboard/custom"
    
    # Define the parameters for the request
    params = {
        "year": "2023",  # Using 2023 as it's the most recent complete season
        "type": "batter",
        "min": "q",  # Qualified batters
        "selections": ",".join(STAT_LABELS.keys()),
        "sort": "xwoba",
        "sortDir": "desc",
        "csv": "true"  # Request CSV format
    }
    
    # Set headers to mimic a browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/csv"
    }
    
    try:
        # Make the request
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Read CSV data
        df = pd.read_csv(io.StringIO(response.text))
        
        # Convert percentage strings to floats
        percent_columns = [col for col in df.columns if '_percent' in col or col.endswith('_rate')]
        for col in percent_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].str.rstrip('%') if df[col].dtype == 'object' else df[col], errors='coerce') / 100
                
        return df
        
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}")
        return None
    except ValueError as e:
        st.error(f"Data parsing error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

def create_correlation_plot(df, x_metric, y_metric):
    """Create an interactive correlation plot using plotly."""
    # Calculate correlation coefficient
    correlation = df[x_metric].corr(df[y_metric])
    
    # Get human-readable labels
    x_label = STAT_LABELS.get(x_metric, x_metric)
    y_label = STAT_LABELS.get(y_metric, y_metric)
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x=x_metric,
        y=y_metric,
        hover_data=['last_name, first_name'],  # Use the actual combined column name
        template='plotly_white',
        title=f'Correlation between {x_label} and {y_label}'
    )
    
    # Update hover template to show player name
    hovertemplate = (
        "<b>%{customdata[0]}</b><br>" +
        f"{x_label}" + ": %{x:.3f}<br>" +
        f"{y_label}" + ": %{y:.3f}<br>" +
        "<extra></extra>"
    )
    fig.update_traces(hovertemplate=hovertemplate)
    
    # Update axis labels
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label
    )
    
    # Add trendline
    z = np.polyfit(df[x_metric], df[y_metric], 1)
    p = np.poly1d(z)
    fig.add_trace(
        go.Scatter(
            x=df[x_metric],
            y=p(df[x_metric]),
            mode='lines',
            name='Trend Line',
            line=dict(color='red', dash='dash')
        )
    )
    
    # Update layout
    fig.update_layout(
        plot_bgcolor='white',
        width=800,
        height=600,
        title={
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    
    return fig, correlation

# Main app logic
def main():
    # Load data
    with st.spinner('Fetching MLB data...'):
        df = fetch_baseball_savant_data()
    
    if df is not None:
        # Get list of available metrics (excluding non-numeric columns)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create options with human-readable labels
        options = [(col, STAT_LABELS.get(col, col)) for col in numeric_columns]
        options.sort(key=lambda x: x[1])  # Sort by human-readable label
        
        # Create two columns for metric selection
        col1, col2 = st.columns(2)
        
        with col1:
            x_metric = st.selectbox(
                'Select X-axis metric',
                options=numeric_columns,
                format_func=lambda x: STAT_LABELS.get(x, x),
                index=numeric_columns.index('xwoba') if 'xwoba' in numeric_columns else 0
            )
            
        with col2:
            y_metric = st.selectbox(
                'Select Y-axis metric',
                options=numeric_columns,
                format_func=lambda x: STAT_LABELS.get(x, x),
                index=numeric_columns.index('batting_avg') if 'batting_avg' in numeric_columns else 0
            )
        
        # Create and display correlation plot
        fig, correlation = create_correlation_plot(df, x_metric, y_metric)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display correlation statistics
        st.subheader("Statistical Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Pearson Correlation Coefficient", f"{correlation:.3f}")
            
        with col2:
            # Calculate R-squared
            r_squared = correlation ** 2
            st.metric("R-squared Value", f"{r_squared:.3f}")
        
        # Add interpretation
        st.markdown("### Interpretation")
        if abs(correlation) > 0.7:
            strength = "strong"
        elif abs(correlation) > 0.3:
            strength = "moderate"
        else:
            strength = "weak"
            
        direction = "positive" if correlation > 0 else "negative"
        
        st.write(f"""
        There is a {strength} {direction} correlation between {STAT_LABELS.get(x_metric, x_metric)} 
        and {STAT_LABELS.get(y_metric, y_metric)}. The R-squared value indicates that 
        {(r_squared * 100):.1f}% of the variance in {STAT_LABELS.get(y_metric, y_metric)} can be 
        explained by {STAT_LABELS.get(x_metric, x_metric)}.
        """)

if __name__ == "__main__":
    main() 