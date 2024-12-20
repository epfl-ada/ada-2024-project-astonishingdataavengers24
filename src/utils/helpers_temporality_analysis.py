import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import ast
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
import dask.dataframe as dd

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

def plot_movies_and_news_frequency(theme, time_unit='Year'):
    """
    Plot the evolution of the frequency of movies and news per decade or year, normalized by the full datasets.
    Arguments:
        df_movie: DataFrame containing movie data.
        theme: String representing the theme (column name in df_news).
        time_unit: String specifying time unit ('Year' or 'Decade').

    Returns:
        Plotly figure.
    """ 
    TIME_START = 1930
    TIME_END = 2013
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
    df_movie = pd.read_csv('../../data/df_movies/cosine_similarity_movies.csv')
    if time_unit == 'Decade':
        df_movie[movie_time_column] = df_movie[movie_time_column].apply(lambda x: x - x % 10)
    # Keep only the date and theme columns
    df_movie = df_movie[[movie_time_column, theme_column]]

    # Load news cosine similarity data
    df_news = pd.read_csv('../../data/df_news/cosine_similarity_news.csv')
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

    title = f"Evolution of Movies and News Frequency in {theme}" + (" by Decade" if time_unit == 'Decade' else  " by Year")
     # Line plot
    fig = px.line(
        evolution,
        x=evolution.index,
        y=['Normalized_Movie_Count', 'Normalized_News_Count'],
        labels={'value': 'Percentage'},
        title=title,
        markers=True
    )

 
    # Use specific colors
    fig.update_traces(name="Movies", selector=dict(name="Normalized_Movie_Count"), line=dict(color=MOVIES_MARKER, width=2))
    fig.update_traces(name="News", selector=dict(name="Normalized_News_Count"), line=dict(color=NEWS_MARKER, width=2))
    fig.update_layout(
        xaxis_title=time_unit,
        yaxis_title=f"Percentage of Movies and News in {theme} Theme",
        legend_title="Source of data",
        xaxis=dict(tickangle=45),
        template="plotly_white"
    )

    # Compute the correlation between movies and news
    correlation = evolution['Normalized_Movie_Count'].corr(evolution['Normalized_News_Count'])
    print(f"Correlation between movies and news for theme '{theme}': {correlation}")

    return fig