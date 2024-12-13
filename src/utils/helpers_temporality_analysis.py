import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import ast

def load_theme_dataset(name_of_csv):
    """ Load the dataset per theme """
    data_folder = '../../data/df_movies/'
    df_plot = pd.read_csv(data_folder + name_of_csv)
    df_plot = df_plot[df_plot['Decade'].notna()]
    df_plot = df_plot[df_plot['Grouped_genres'] != '[nan]']
    return df_plot

def plot_movie_frequency_by_decade(df):
    """
    Plot the evolution of the frequency of movies per decade.

    Arguments:
        df: the DataFrame containing movie data.
    """
    # Count number of movies per decade
    movie_evolution = df.groupby('Decade').size().reset_index(name='Count')
    
    # Line plot
    fig = px.line(
        movie_evolution,
        x='Decade',
        y='Count',
        title='Evolution of Movie Frequency by Decade',
        markers=True
    )
    
    # Layout
    fig.update_traces(line_color='blue')
    fig.update_layout(xaxis_title="Decade", yaxis_title="Number of Movies", xaxis=dict(tickangle=45), template="plotly_white")
    fig.show()

def plot_movies_and_news_frequency_by_decade(df_movie, df_news):
    """
    Plot the evolution of the frequency of movies and news per decade.

    Arguments:
        df_movie: the DataFrame containing movie data.
        df_news: the DataFrame containing news data.
    """
    # Count number of movies per decade
    movie_evolution = df_movie.groupby('Decade').size().reset_index(name='Movie_Count')
    
    # Count number of news articles per decade
    news_evolution = df_news.groupby('Decade').size().reset_index(name='News_Count')
    
    # Merge the two DataFrames on the 'Decade' column
    evolution = pd.merge(movie_evolution, news_evolution, on='Decade', how='outer').fillna(0)
    
    # Create the plot
    fig = px.line(
        evolution,
        x='Decade',
        y=['Movie_Count', 'News_Count'],
        title='Evolution of Movie and News Frequency by Decade',
        markers=True
    )
    
    # Layout
    fig.update_traces(line_color=['blue', 'red'])
    fig.update_layout(
        xaxis_title="Decade",
        yaxis_title="Count",
        xaxis=dict(tickangle=45),
        template="plotly_white"
    )
    
    # Show the plot
    fig.show()
