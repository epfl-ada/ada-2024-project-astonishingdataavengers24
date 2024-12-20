import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import ast
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
import dask.dataframe as dd
from scipy.signal import correlate

TIME_START = 1930
TIME_END = 2015

def load_theme_dataset(name_of_csv):
    """ 
    Load the dataset per theme 
    
    Arugments:
        name_of_csv: String representing the name of the file to load.
    """
    data_folder = '../../data/df_movies/'
    df_plot = pd.read_csv(data_folder + name_of_csv)
    df_plot = df_plot[df_plot['Decade'].notna()]
    df_plot = df_plot[df_plot['Movie_release_date'].notna()]
    df_plot = df_plot[df_plot['Grouped_genres'] != '[nan]']
    return df_plot

MOVIES_MARKER = px.colors.qualitative.Prism[1]  # Blue
NEWS_MARKER = px.colors.qualitative.Prism[8]   # Purple
    
def plot_movie_frequency(df, theme, time_unit='Year'):
    """
    Plot the evolution of the frequency of movies by either year or decade, normalized by full dataset.

    Arguments:
        df: the DataFrame containing movie data.
        theme: String representing the theme.
        time_unit: String specifying time unit ('Year' or 'Decade).
        
    Returns:
        Plotly figure.
    """
    if time_unit == 'Year':
        time_column = 'Movie_release_date'
    elif time_unit == 'Decade':
        time_column = 'Decade'
    else:
        raise ValueError("Invalid time_unit. Please choose either 'Year' or 'Decade'.")

    # Count number of movies per year/decade
    movie_evolution = df.groupby(time_column).size().reset_index(name='Count')

    # Upload full dataset for normalization
    df_full = pd.read_csv('../../data/MovieSummaries/movies_metadata_cleaned.csv')
    df_full = df_full[df_full[time_column].notna()]  # Drop empty time units

    # Count total number of movies per year/decade in the full dataset
    total_per_time_unit = df_full.groupby(time_column).size().reset_index(name='Total')

    # Merge and calculate normalized count
    movie_evolution = movie_evolution.merge(total_per_time_unit, on=time_column, how='left')
    movie_evolution['Normalized_Count'] = movie_evolution['Count'] / movie_evolution['Total']
    
    # Line plot
    fig = px.line(
        movie_evolution,
        x=time_column,
        y='Normalized_Count',
        title=f'Evolution of Movie Frequency by {time_unit}',
        markers=True,
        hover_data={
            time_column: True,
            'Normalized_Count': ':.2f%'
        },
        labels={
            time_column: time_unit,
            'Normalized_Count': f'Percentage of movies about {theme} theme in the dataset'
        }
    )
    
    # Layout
    fig.update_traces(line_color=MOVIES_MARKER)
    fig.update_layout(
        xaxis_title=time_unit,
        yaxis_title=f"Percentage of movies about {theme} theme in the dataset",
        xaxis=dict(tickangle=45),
        template="plotly_white"
    )
    
    return fig

def plot_movies_and_news_frequency(theme, data_folder='../../data/', time_unit='Year',):
    """
    Plot the evolution of the frequency of movies and news per decade or year, normalized by the full datasets.
    Arguments:
        df_movie: DataFrame containing movie data.
        theme: String representing the theme (column name in df_news).
        time_unit: String specifying time unit ('Year' or 'Decade').

    Returns:
        Plotly figure.
    """ 
    MOVIES_MARKER = px.colors.qualitative.Prism[1]  # Blue
    NEWS_MARKER = px.colors.qualitative.Prism[8]   # Purple

  
    movie_time_column = 'year'
    news_time_column = 'year'

    if time_unit == 'Year':
        time_range = pd.DataFrame({movie_time_column: range(TIME_START, TIME_END+1)})
    elif time_unit == 'Decade':
        time_range = pd.DataFrame({movie_time_column: range(TIME_START, TIME_END + 6 , 10)})
    else:
        raise ValueError("Invalid time_unit. Please choose either 'Year' or 'Decade'.")

    theme_mapping = {
        'Technology': 'technology',
        'Cold War': 'cold_war',
        'Economy': 'economy',
        'Health': 'health',
        'Gender Equality': 'gender_equality',
        'Migration': 'migration',
        'Economic Crisis': 'economy',
        'Vietnam War': 'vietnam',
        'World War II': 'ww2'
    }
    if theme not in theme_mapping:
        raise ValueError("Invalid theme. Please choose a valid theme.")

    theme_column = theme_mapping[theme]

    # Load movie cosine similarity data
    print(os.getcwd(), os.listdir())
    df_movie = pd.read_csv(os.path.join(data_folder, 'df_movies/cosine_similarity_movies.csv'))
    if time_unit == 'Decade':
        df_movie[movie_time_column] = df_movie[movie_time_column].apply(lambda x: x - x % 10)
    # Keep only the date and theme columns
    df_movie = df_movie[[movie_time_column, theme_column]]

    # Load news cosine similarity data
    df_news = pd.read_csv(os.path.join(data_folder, 'df_news/cosine_similarity_news.csv'))
    if time_unit == 'Decade':
        df_news[news_time_column] = df_news[news_time_column].apply(lambda x: x - x % 10)
    # Keep only the id, date and theme columns
    df_news = df_news[[news_time_column, theme_column]]

    # Cast cosine similarity to boolean
    df_movie[theme_column] = df_movie[theme_column] > 0.2
    df_news[theme_column] = df_news[theme_column] > 0.2

    # Compute the number of movies and news per time unit
    movies_per_time = df_movie.groupby(movie_time_column).size()
    news_per_time = df_news.groupby(news_time_column).size()

    # Count the number of movies and news per time unit matching the theme
    movies_per_time_theme = df_movie.groupby([movie_time_column]).sum()
    news_per_time_theme = df_news.groupby([news_time_column]).sum()
    # Add missing years
    movies_per_time_theme = movies_per_time_theme.reindex(time_range[movie_time_column], fill_value=0)
    news_per_time_theme = news_per_time_theme.reindex(time_range[movie_time_column], fill_value=0)

    # Filter years outside the range
    movies_per_time_theme = movies_per_time_theme[(movies_per_time_theme.index >= TIME_START) & (movies_per_time_theme.index <= TIME_END)]
    news_per_time_theme = news_per_time_theme[(news_per_time_theme.index >= TIME_START) & (news_per_time_theme.index <= TIME_END)]

    # Normalize by the total number of movies and news
    movies_per_time_theme = movies_per_time_theme.div(movies_per_time, axis=0)
    news_per_time_theme = news_per_time_theme.div(news_per_time, axis=0)

    # Combine the dataframes
    evolution = pd.concat([movies_per_time_theme, news_per_time_theme], axis=1)
    evolution.columns = ['Normalized_Movie_Count', 'Normalized_News_Count']

    # Compute the correlation between movies and news
    correlation = evolution['Normalized_Movie_Count'].corr(evolution['Normalized_News_Count'])
    print(f"Correlation between movies and news for theme '{theme}': {correlation}")

    # Compute the correlation between movies and news
    movies_data = np.array(movies_per_time_theme[theme_column])
    news_data = np.array(news_per_time_theme[theme_column])
    # Replace NaN values with 0
    movies_data[np.isnan(movies_data)] = 0
    news_data[np.isnan(news_data)] = 0
    compute_cross_correlation(movies_data, news_data)

    return

def compute_cross_correlation(movies_freq, news_freq):

    years = np.arange(TIME_START, TIME_END + 1)
    max_lag = (2014 - 1930) // 2  # Maximum lag value 
    # Standardize the series
    movies_freq = (movies_freq - np.mean(movies_freq)) / np.std(movies_freq)
    news_freq = (news_freq - np.mean(news_freq)) / np.std(news_freq)
    
    # Compute cross-correlation
    lags = np.arange(-max_lag, max_lag + 1)
    cross_corr = correlate(movies_freq, news_freq, mode='full')
    # Keep only the cross-correlation values for the lags
    cross_corr = cross_corr[(len(movies_freq) ) - max_lag:(len(movies_freq) ) + max_lag + 1]

    # Find the lag with maximum correlation
    max_corr_idx = np.argmax(cross_corr)
    optimal_lag = lags[max_corr_idx]
    print(f"Optimal lag: {optimal_lag} years")
   
    # Plot cross-correlation
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=lags, y=cross_corr, mode='lines', name='Cross-correlation'))
    fig1.add_shape(
        dict(
            type="line",
            x0=optimal_lag,
            y0=min(cross_corr),
            x1=optimal_lag,
            y1=max(cross_corr),
            line=dict(color="red", width=2)

        )
    )
    
    # Add text annotation with optimal lag
    fig1.add_annotation(
        x=optimal_lag,
        y=max(cross_corr),
        text=f"Optimal lag: {optimal_lag} years",
        showarrow=True,
        arrowhead=1,
        ax=50,
        ay=-50,
        
        bgcolor="white",
        opacity=0.8

    )
    fig1.update_layout(
        title="Cross-correlation between Movies and News",
        xaxis_title="Lag (years)",
        yaxis_title="Cross-correlation",
        template="plotly_white"
    )
    fig1.show()


    # Make interactive plot with slider
    # Create figure
    fig2 = go.Figure()

    # Add traces for each slider step (different lags)
    for lag in range(-max_lag, max_lag +1):  
        shifted_news_freq = np.roll(news_freq, lag)  # Shift news frequency by lag
        fig2.add_trace(
            go.Scatter(
                visible=False,
                mode='lines+markers',
                name=f"News Frequency (Shifted by {lag} years)",
                x=years,
                y=shifted_news_freq,
                line=dict(color=NEWS_MARKER, width=2)
            )
        )

    # Add movies trace (remains visible across all steps)
    fig2.add_trace(
        go.Scatter(
            visible=True,
            mode='lines+markers',
            name="Movies Frequency",
            x=years,
            y=movies_freq,
            line=dict(color=MOVIES_MARKER, width=2)
        )
    )

    # Make one trace for news visible by default
    fig2.data[max_lag].visible = True

    # Create and add slider
    steps = []
    for i in range(len(fig2.data) - 1):  # Exclude Series 1 trace from slider steps
        step = dict(
            method="update",
            args=[
                {"visible": [i == j for j in range(len(fig2.data) - 1)] + [True]},  # Show one trace + Series 1
            ],
        )
        steps.append(step)

    sliders = [dict(
        active=max_lag,  # Default active step (lag = 0)
        currentvalue={"prefix": "Lag: "},
        pad={"t": 50},
        steps=steps
    )]

    # Update layout with slider
    fig2.update_layout(
        sliders=sliders,
        title="Interactive Lag Visualization",
        xaxis_title="Year",
        yaxis_title="Proportion",
        template="plotly_white"
    )

    fig2.show()

    return 
   
