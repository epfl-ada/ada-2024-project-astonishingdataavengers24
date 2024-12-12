""" 

Imports 

"""
import numpy as np
import pandas as pd
import plotly.express as px
import sys
import os
import ast
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from src.utils.helpers_sentiment_analysis import *


""" 

Helpers 

"""
def load_theme_dataset(name_of_csv):
    """ Load the dataset per theme """
    data_folder = '../../src/df_movies_themes/df_movies/'
    df_plot = pd.read_csv(data_folder + name_of_csv)
    df_plot = df_plot[df_plot['Decade'].notna()]
    df_plot = df_plot[df_plot['Grouped_genres'] != '[nan]']
    return df_plot


""" 

Genre Analysis 

"""

def plot_movie_frequency_by_decade():
    """ Plot the frequency of movies by decade """
    return 

def plot_top_movie_genres():
    """ Plot the top movie genres """
    return

def plot_top_movie_genres_by_decade():
    """ Plot the top movie genres per decade """
    return

""" 

Emotions Analysis 

"""
def create_emotions_column(df_movies):
    df_movies = df_movies.copy()

    # Apply VADER Sentiment Analysis
    df_movies.loc[:, 'VADER_Sentiment'] = df_movies['plot_summary'].apply(get_vader_sentiment)

    # Apply Emotion Analysis
    df_movies.loc[:, 'Emotions'] = df_movies['plot_summary'].apply(extract_emotions)

    # Classify Sentiment Based on Emotions
    df_movies.loc[:, 'Emotions_Sentiment'] = df_movies['Emotions'].apply(classify_sentiment_from_emotions)
    df_movies['Emotions']
    return df_movies

def emotions_group():
    positive_emotions = ['positive', 'anticipation', 'surprise', 'joy', 'trust']
    negative_emotions = ['negative', 'anger', 'fear', 'disgust', 'sadness']
    return (positive_emotions, negative_emotions)
   
def plot_emotion_counts(df, theme):
    """ Plot the count of emotions across movies """
 
    def aggregate_emotions(emotions_series):
        """
        Aggregates the counts of each emotion from the 'Emotions' column.
        """
        aggregated_emotions = {}
        for emotions_dict in emotions_series:
            if isinstance(emotions_dict, dict):
                for emotion, count in emotions_dict.items():
                    aggregated_emotions[emotion] = aggregated_emotions.get(emotion, 0) + count
        return aggregated_emotions


    emotion_totals = aggregate_emotions(df['Emotions'])

    emotion_totals_df = pd.DataFrame(list(emotion_totals.items()), columns=['Emotion', 'Count'])

    fig = px.bar(
        emotion_totals_df,
        x='Emotion',
        y='Count',
        color='Emotion',
        title=f"Total Counts of Emotions Across Movies for theme: {theme}",
        labels={'Emotion': 'Emotion', 'Count': 'Count'},
        text='Count', 
        #color_discrete_sequence=px.colors.qualitative.Set1 CHANGE COLOR IF NEEDED 
    )
    fig.update_layout(xaxis_title="Emotion", yaxis_title="Total Count")
    fig.show()
    
    return

def plot_emotion_counts_by_decade(movie_df, theme):
    """
    Plot the count of emotions across movies by decade.

    Arguments:
        - movie_df: DataFrame of movies with a 'Decade' column.
    """
    def aggregate_emotions_by_decade(df):
        """
        Aggregates the counts of each emotion by decade.
        """
        aggregated_emotions = []
        for _, row in df.iterrows():
            if isinstance(row['Emotions'], dict):
                for emotion, count in row['Emotions'].items():
                    aggregated_emotions.append({
                        'Decade': row['Decade'],
                        'Emotion': emotion,
                        'Count': count
                    })
        return pd.DataFrame(aggregated_emotions)

    # Aggregate emotions by decade
    emotions_by_decade_df = aggregate_emotions_by_decade(movie_df)

    # Summarize counts by decade and emotion
    emotions_summary = emotions_by_decade_df.groupby(['Decade', 'Emotion'])['Count'].sum().reset_index()

    # Plot the line chart
    fig = px.line(
        emotions_summary,
        x='Decade',
        y='Count',
        color='Emotion',
        title=f"Counts of Emotions by Decade for {theme}",
        labels={'Decade': 'Decade', 'Count': 'Count', 'Emotion': 'Emotion'},
        markers=True
    )
    fig.update_layout(
        xaxis_title="Decade",
        yaxis_title="Total Emotion Count",
        xaxis=dict(dtick=10)
    )
    fig.show()


def plot_emotion_sentiment_counts(movie_df, theme):
    """
    Plots a bar chart showing the count of emotions categorized as sentiments (positive or negative).

    Arguments:
        - movie_df: DataFrame of movies.
        - theme: The fixed theme (e.g., "War").
    """
    # Define emotion groups
    positive_emotions = ['positive', 'anticipation', 'surprise', 'joy', 'trust']
    negative_emotions = ['negative', 'anger', 'fear', 'disgust', 'sadness']

    # Ensure the genres column is parsed correctly
    movie_df = movie_df.copy()
    movie_df['Movie_genres'] = movie_df['Movie_genres'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # Expand the 'Emotions' dictionary into individual rows
    emotion_data = []
    for _, row in movie_df.iterrows():
        if isinstance(row['Emotions'], dict):
            for emotion, count in row['Emotions'].items():
                # Classify emotions as positive or negative
                if emotion in positive_emotions:
                    emotion_data.append({
                        'Theme': theme,
                        'Sentiment': 'Positive',
                        'Emotion': emotion,
                        'Emotion_Count': count
                    })
                elif emotion in negative_emotions:
                    emotion_data.append({
                        'Theme': theme,
                        'Sentiment': 'Negative',
                        'Emotion': emotion,
                        'Emotion_Count': count
                    })

    # Create a new DataFrame from the expanded data
    df_emotion_expanded = pd.DataFrame(emotion_data)

    # Aggregate counts by sentiment
    sentiment_counts = df_emotion_expanded.groupby('Sentiment')['Emotion_Count'].sum().reset_index()

    # Create the bar chart
    fig = px.bar(
        sentiment_counts,
        x='Sentiment',
        y='Emotion_Count',
        color='Sentiment',
        title=f"Emotion Counts Categorized by Sentiment for Theme: {theme}",
        labels={'Sentiment': 'Sentiment', 'Emotion_Count': 'Total Emotion Count'},
        text='Emotion_Count'
    )

    # Update layout for better visualization
    fig.update_layout(xaxis_title="Sentiment", yaxis_title="Total Emotion Count")
    
    # Show the chart
    fig.show()
    
""" 

Sentiment Analysis 

"""

def plot_sentiment_distribution_nltk():
    """ Plot the sentiment distribution using the NLTK library """
    return

def plot_sentiment_distribution_textblob():
    """ Plot the sentiment distribution using the TextBlob library """
    return

def plot_sentiment_distribution_vader():
    """ Plot the sentiment distribution using the VADER library """
    return

def plot_sentiment_distribution_huggingface():
    """ Plot the sentiment distribution using the Hugging Face library """
    return

def plot_combined_sentiment_distribution():
    """ Plot the combined sentiment distribution across all libraries """
    return 

def plot_top_movie_genres_by_sentiment():
    """ Plot the top movie genres categorized by sentiment """
    return

def plot_sentiment_distribution_by_decade():
    """ Plot the sentiment distribution by decade """
    return

def plot_sunburst_genres_sentiment_emotions(movie_df, theme, top_n=5):
    """
    Plots a sunburst chart with the hierarchy: Theme → Top Genres → Sentiment → Emotions.

    Arguments:
        - movie_df: DataFrame of movies.
        - theme: The fixed theme.
        - top_n: Number of top genres to include in the chart (default is 10).
    """

    (positive_emotions, negative_emotions) = emotions_group()

    movie_df = movie_df.copy()
    
    # Get the movie genres
    movie_df['Movie_genres'] = movie_df['Movie_genres'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    df_exploded = movie_df.explode('Movie_genres')
    
    top_genres = (
        df_exploded['Movie_genres']
        .value_counts()
        .nlargest(top_n)
        .index
        .tolist()
    )

    # Filter for rows with the top genres 
    df_exploded = df_exploded[df_exploded['Movie_genres'].isin(top_genres)]

    # Expand the dictionary into individual rows
    emotion_data = []
    for _, row in df_exploded.iterrows():
        if isinstance(row['Emotions'], dict):
            for emotion, count in row['Emotions'].items():
                if row['Emotions_Sentiment'] == 'POSITIVE' and emotion in positive_emotions:
                    emotion_data.append({
                        'Theme': theme,
                        'Genre': row['Movie_genres'],
                        'Sentiment': row['Emotions_Sentiment'],
                        'Emotion': emotion,
                        'Emotion_Count': count
                    })
                elif row['Emotions_Sentiment'] == 'NEGATIVE' and emotion in negative_emotions:
                    emotion_data.append({
                        'Theme': theme,
                        'Genre': row['Movie_genres'],
                        'Sentiment': row['Emotions_Sentiment'],
                        'Emotion': emotion,
                        'Emotion_Count': count
                    })

    df_emotion_expanded = pd.DataFrame(emotion_data)

    # Filter out rows where the emotion count is zero
    df_emotion_expanded = df_emotion_expanded[df_emotion_expanded['Emotion_Count'] > 0]

    
    # Plot
    fig = px.sunburst(
        df_emotion_expanded,
        path=['Theme', 'Genre', 'Sentiment', 'Emotion'],  # Define hierarchy
        values='Emotion_Count',  
        title=f"Overall chart for {theme}",
        labels={
            'Theme': 'Theme',
            'Genre': 'Genre',
            'Sentiment': 'Sentiment',
            'Emotion': 'Emotion',
            'Emotion_Count': 'Emotion Count'
        },
        color='Emotion_Count', 
        color_continuous_scale='RdBu'
    )

    fig.show()

""" 

Temporal Analysis 

"""
def plot_movies_and_news_frequency_by_decade():
    """ Plot the frequency of movies and news by decade """
    return
