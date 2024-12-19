import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import ast

import dask.dataframe as dd

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

def plot_movies_and_news_frequency(df_movie, theme, time_unit='Decade'):
    """
    Plot the evolution of the frequency of movies and news per decade or year, normalized by the full datasets.

    Arguments:
        df_movie: DataFrame containing movie data.
        theme: String representing the theme (column name in df_news).
        time_unit: String specifying time unit ('Year' or 'Decade').
        
    Returns:
        Plotly figure.
    """

    df_movie = df_movie[(df_movie['Movie_release_date'] >= 1965) & (df_movie['Movie_release_date'] <= 2015)]

    if time_unit == 'Year':
        movie_time_column = 'Movie_release_date'
        news_time_column = 'year'
        time_range = pd.DataFrame({movie_time_column: range(1965, 2016)})
    elif time_unit == 'Decade':
        movie_time_column = 'Decade'
        news_time_column = 'decade'
        time_range = pd.DataFrame({movie_time_column: range(1960, 2021, 10)})
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

    # Count the number of movies per time unit
    movie_evolution = df_movie.groupby(movie_time_column).size().reset_index(name='Movie_Count')

    # Filter news data for the selected theme and count the number of news articles per time unit
    df_news = pd.read_csv('../../data/df_news/cosine_similarity_news_cleaned.csv')
    df_news_theme = df_news[df_news[theme_column] == 1]
    news_evolution = df_news_theme.groupby(news_time_column).size().reset_index(name='News_Count')
    
    # Load full datasets for normalization
    df_full_movies = pd.read_csv('../../data/MovieSummaries/movies_metadata_cleaned.csv')
    df_full_movies = df_full_movies[(df_full_movies['Movie_release_date'] >= 1965) & (df_full_movies['Movie_release_date'] <= 2015)]
    total_movies = df_full_movies.groupby(movie_time_column).size().reset_index(name='Total_Movies')
    total_news = df_news.groupby(news_time_column).size().reset_index(name='Total_News')

    # Merge movie data with total movie data for normalization
    movie_evolution = movie_evolution.merge(total_movies, on=movie_time_column, how='left')
    movie_evolution['Normalized_Movie_Count'] = movie_evolution['Movie_Count'] / movie_evolution['Total_Movies']

    # Merge news data with total news data for normalization
    news_evolution = news_evolution.merge(total_news, on=news_time_column, how='left')
    news_evolution['Normalized_News_Count'] = news_evolution['News_Count'] / news_evolution['Total_News']

    # Merge movies and news data
    evolution = pd.merge(movie_evolution, news_evolution, left_on=movie_time_column, right_on=news_time_column, how='outer')

    # Ensure all time intervals are represented
    evolution = time_range.merge(evolution, on=movie_time_column, how='left')
    evolution.fillna(0, inplace=True)

    # Line plot
    fig = px.line(
        evolution,
        x=movie_time_column,
        y=['Normalized_Movie_Count', 'Normalized_News_Count'],
        title=f'Evolution of Movie and News Frequency about {theme} by {time_unit}',
        labels={movie_time_column: time_unit, 'value': 'Percentage'},
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
        
    return fig
      





