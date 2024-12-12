""" 

Imports 

"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import ast
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from src.utils.helpers_sentiment_analysis import *

# import matplotlib.pyplot as plt
# import seaborn as sns

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from transformers import pipeline
from nrclex import NRCLex


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

def plot_overall_top_genres(df, x=10):
    """
    Plot a pie chart for the top x genres overall.
    
    Arguments:
        df: the DataFrame containing movie data.
        x: the number of top genres to display.
    """
    # Group by 'Grouped_genres' and count occurrences
    df['Grouped_genres'] = df['Grouped_genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Exploding the 'Grouped_genres' column so that each genre becomes a separate row
    df_expanded = df.explode('Grouped_genres')

    # Group by 'Grouped_genres' and count occurrences
    genre_counts = df_expanded.groupby(['Grouped_genres']).size().reset_index(name="Count")
    genre_counts = genre_counts.sort_values(by=['Count'], ascending=False)

    # Get the top x genres overall
    top_x_genre_overall = genre_counts.head(x)

    # Create an interactive pie chart using Plotly
    fig = px.pie(
        top_x_genre_overall, 
        values='Count', 
        names='Grouped_genres',
        title=f'Top {x} Movie Genres Overall',
        labels={
            'Grouped_genres': 'Genre',
            'Count': 'Count'
        },
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    # Customize layout
    fig.update_traces(textinfo='percent+label', pull=[0.1 if i == 0 else 0 for i in range(x)])
    fig.update_layout(height=600, width=600)

    # Show the plot
    fig.show()

def plot_top_genres_by_decade(df, x=10):
    """
    Plot the top x genres per decade for each genre.

    Arguments:
        df: the DataFrame containing movie data.
        x: the number of top genres per decade.
    """
    # Ensure the 'Grouped_genres' column is properly evaluated if it's a string representation of a list
    df['Grouped_genres'] = df['Grouped_genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Exploding the 'Grouped_genres' column so that each genre becomes a separate row
    df_expanded = df.explode('Grouped_genres')

    # Group by 'Decade' and 'Grouped_genres' and count occurrences
    genre_distribution = df_expanded.groupby(['Decade', 'Grouped_genres']).size().reset_index(name="Count")

    # Sort 'Decade' and 'Count'
    genre_distribution = genre_distribution.sort_values(by=['Decade', 'Count'], ascending=[True, False])

    # Top x genres per decade
    top_x_genres_per_decade = genre_distribution.groupby(['Decade']).head(x)

    # Avoid warning using copy
    top_x_genres_per_decade = top_x_genres_per_decade.copy()

    # Normalize the counts by calculating the percentage for each genre
    decade_totals = top_x_genres_per_decade.groupby('Decade')['Count'].transform('sum')
    top_x_genres_per_decade['Percentage'] = (top_x_genres_per_decade['Count'] / decade_totals) * 100

    # Pivot the data for Plotly
    top_genres_normalized = top_x_genres_per_decade.pivot(index='Decade', columns='Grouped_genres', values='Percentage').fillna(0)

    # Color palette
    unique_genres = top_genres_normalized.columns
    colors = sns.color_palette("Spectral", n_colors=len(unique_genres)) 
    genre_colors = dict(zip(unique_genres, [f"rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, 0.8)" for c in colors]))

    # Create the stacked bar chart with Plotly
    fig = go.Figure()

    for genre in top_genres_normalized.columns:
        fig.add_trace(go.Bar(
            x=top_genres_normalized.index,
            y=top_genres_normalized[genre],
            name=genre,
            marker_color=genre_colors[genre]
        ))

    # Update layout
    fig.update_layout(
        title=f'Top {x} Movie Genres By Decade',
        xaxis=dict(title='Decade'),
        yaxis=dict(title='Percentage of Total Genres (%)', ticksuffix='%'),
        barmode='stack',
        legend=dict(title='Genres'),
        template='plotly_white',
        height=600,
        width=900
    )

    # Show the plot
    fig.show()

""" 

Emotions Analysis 

"""
def create_emotions_column(df):
    df = df.copy()

    # Apply VADER Sentiment Analysis
    df.loc[:, 'VADER_Sentiment'] = df['plot_summary'].apply(get_vader_sentiment)

    # Apply Emotion Analysis
    df.loc[:, 'Emotions'] = df['plot_summary'].apply(extract_emotions)

    # Classify Sentiment Based on Emotions
    df.loc[:, 'Emotions_Sentiment'] = df['Emotions'].apply(classify_sentiment_from_emotions)
    df['Emotions']
    return df

def emotions_group():
    positive_emotions = ['positive', 'anticipation', 'surprise', 'joy', 'trust']
    negative_emotions = ['negative', 'anger', 'fear', 'disgust', 'sadness']
    return (positive_emotions, negative_emotions)
   
def plot_emotion_counts(df, theme):
    """ 
    Plot the count of emotions across movies.
    
    Arguments:
        df: the DataFrame containing movie data.
        theme: theme of the movies in the dataframe.
    """
 
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

def plot_emotion_counts_by_decade(df, theme):
    """
    Plot the count of emotions across movies by decade.

    Arguments:
        df: the DataFrame containing movie data.
        theme: theme of the movies in the dataframe.
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
    emotions_by_decade_df = aggregate_emotions_by_decade(df)

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


def plot_emotion_sentiment_counts(df, theme):
    """
    Plots a bar chart showing the count of emotions categorized as sentiments (positive or negative).

    Arguments:
        df: the DataFrame containing movie data.
        theme: theme of the movies in the dataframe.
    """
    # Define emotion groups
#     positive_emotions = ['positive', 'anticipation', 'surprise', 'joy', 'trust']
#     negative_emotions = ['negative', 'anger', 'fear', 'disgust', 'sadness']

    positive_emotions, negative_emotions = emotions_group()

    # Ensure the genres column is parsed correctly
    df = df.copy()
    df['Movie_genres'] = df['Movie_genres'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # Expand the 'Emotions' dictionary into individual rows
    emotion_data = []
    for _, row in df.iterrows():
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
def preprocess_text(text):
    """Preprocess text: tokenization, lemmatization"""
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def get_nltk_sentiment(text):
    """NLTK Sentiment Analysis using VADER"""
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return "POSITIVE"
    elif scores['compound'] <= -0.05:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def get_textblob_sentiment(text):
    """TextBlob Sentiment Analysis"""
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Returns a polarity score (-1 to 1)

def get_vader_sentiment(text):
    """VADER Sentiment Analysis"""
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)['compound']

def extract_emotions(text):
    """Emotion analysis using NRC Lexicon"""
    try:
        emotion = NRCLex(text)
        return emotion.raw_emotion_scores
    except Exception as e:
        return str(e)

def classify_sentiment_from_emotions(emotions):
    """Classify sentiment based on emotions"""
#     positive_emotions = ['positive', 'anticipation', 'surprise', 'joy', 'trust']
#     negative_emotions = ['negative', 'anger', 'fear', 'disgust', 'sadness']
    positive_emotions, negative_emotions = emotions_group()
    positive_score = sum(emotions.get(emotion, 0) for emotion in positive_emotions)
    negative_score = sum(emotions.get(emotion, 0) for emotion in negative_emotions)
    return 'POSITIVE' if positive_score > negative_score else 'NEGATIVE'

def huggingface_sentiment_analysis(texts, model_name="distilbert-base-uncased-finetuned-sst-2-english", batch_size=16):
    """Hugging Face Transformer Sentiment Analysis in batches"""
    sentiment_analyzer = pipeline("sentiment-analysis", model=model_name, truncation=True)
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results.extend(sentiment_analyzer(batch))
    return results

def perform_sentiment_analysis(df):
    """
    Perform various sentiment and emotion analyses on the 'plot_summary' column of a dataframe.

    Arguments:
        df: the DataFrame containing movie data.

    Returns:
        pd.DataFrame: updated dataframe with new sentiment analysis columns.
    """
    # Remove the rows without decades and genres
    df = df[df['Decade'].notna()]
    df = df[df['Grouped_genres'] != '[nan]']
    df = df.copy()

    # Preprocessing (put in preprocessing?)
    df.loc[:, 'plot_summary'] = df['plot_summary'].apply(preprocess_text)

    # NLTK Sentiment Analysis
    df.loc[:, 'NLTK_Sentiment'] = df['plot_summary'].apply(get_nltk_sentiment)

    # TextBlob Sentiment Analysis
    df.loc[:, 'TB_Sentiment'] = df['plot_summary'].apply(get_textblob_sentiment)

    # VADER Sentiment Analysis
    df.loc[:, 'VADER_Sentiment'] = df['plot_summary'].apply(get_vader_sentiment)

    # Emotion Analysis with NRC Lexicon
    df.loc[:, 'Emotions'] = df['plot_summary'].apply(extract_emotions)

    # Sentiment from Emotions
    df.loc[:, 'Emotions_Sentiment'] = df['Emotions'].apply(classify_sentiment_from_emotions)

    # Huggingface Sentiment
    # batch_results = huggingface_sentiment_analysis(df['plot_summary'].tolist())
    # df.loc[:, 'HFT_Sentiment_Result'] = batch_results
    # df.loc[:, 'HFT_Sentiment'] = df['HFT_Sentiment_Result'].apply(lambda x: x['label'])
    # df.loc[:, 'HFT_Score'] = df['HFT_Sentiment_Result'].apply(lambda x: x['score'])

    return df

def plot_sentiment_by_decade(df, technique='NLTK'):
    """
    Plot Sentiment Evolution (Positive/Negative) over Decades for a specified sentiment analysis technique.
    
    Arguments:
        df: the DataFrame containing movie data.
        technique (str): sentiment analysis technique to plot ('NLTK', 'TextBlob', 'VADER', 'Emotions', 'HuggingFace').
    """
    # Ensure given technique is valid
    techniques = ['NLTK', 'TextBlob', 'VADER', 'Emotions', 'HuggingFace']
    if technique not in techniques:
        raise ValueError(f"Invalid technique. Choose from: {', '.join(techniques)}")

    positive_label = 'POSITIVE'
    negative_label = 'NEGATIVE'

    if technique == 'NLTK':
        sentiment_col = 'NLTK_Sentiment'
    elif technique == 'TextBlob':
        sentiment_col = 'TB_Sentiment'
    elif technique == 'VADER':
        sentiment_col = 'VADER_Sentiment'
    elif technique == 'Emotions':
        sentiment_col = 'Emotions_Sentiment'
    elif technique == 'HuggingFace':
        sentiment_col = 'HFT_Sentiment'

    if technique in ['TextBlob', 'VADER']:
        # Use numerical sentiment values for TextBlob and VADER
        sentiment_counts = df.groupby('Decade')[sentiment_col].apply(lambda x: (x > 0).sum()).reset_index(name='POSITIVE')
        sentiment_counts['NEGATIVE'] = df.groupby('Decade')[sentiment_col].apply(lambda x: (x < 0).sum()).reset_index(name='NEGATIVE')['NEGATIVE']
#         sentiment_counts = df.groupby('Decade')[sentiment_col].apply(lambda x: (x < 0).sum()).reset_index(name='NEGATIVE')
    else:
        # Use positive and negative labels for NLTK, Emotions and HuggingFace
        sentiment_counts = df.groupby('Decade')[sentiment_col].value_counts().unstack(fill_value=0).reset_index()
        sentiment_counts.rename(columns={positive_label: 'POSITIVE', negative_label: 'NEGATIVE'}, inplace=True)

    # Normalize counts
    sentiment_counts['Total'] = sentiment_counts['POSITIVE'] + sentiment_counts['NEGATIVE']
    sentiment_counts['Positive_Percentage'] = (sentiment_counts['POSITIVE'] / sentiment_counts['Total']) * 100
    sentiment_counts['Negative_Percentage'] = (sentiment_counts['NEGATIVE'] / sentiment_counts['Total']) * 100

    # Stacked bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=sentiment_counts['Decade'],
        y=sentiment_counts['Positive_Percentage'],
        name='Positive sentiment',
        marker_color='blue'
    ))

    fig.add_trace(go.Bar(
        x=sentiment_counts['Decade'],
        y=sentiment_counts['Negative_Percentage'],
        name='Negative sentiment',
        marker_color='orange'
    ))
    
    fig.update_layout(
        title=f'{technique} Sentiment Evolution by Decade',
        xaxis=dict(title='Decade'),
        yaxis=dict(title='Percentage of Sentiment', ticksuffix='%'),
        barmode='stack',
        legend=dict(title='Sentiment'),
        template='plotly_white'
    )

    # Show the plot
    fig.show()
    
def plot_combined_sentiment_by_decade(df):
    """
    Plot combined Sentiment Evolution (Positive/Negative) over Decades by aggregating sentiment across all techniques.
    
    Arguments:
        df: the DataFrame containing movie data.
    """
    # Define the sentiment columns for each technique
    techniques = {
        'NLTK': 'NLTK_Sentiment',
        'TextBlob': 'TB_Sentiment',
        'VADER': 'VADER_Sentiment',
        'Emotions': 'Emotions_Sentiment'
    }
    
    # Columns for counts of total positive and negative sentiments by decade
    sentiment_counts = pd.DataFrame(index=df['Decade'].unique())

    # Initialize columns for positive and negative counts
    sentiment_counts['POSITIVE'] = 0
    sentiment_counts['NEGATIVE'] = 0

    for technique, sentiment_col in techniques.items():
        if technique in ['TextBlob', 'VADER']:
            # Use numerical sentiment values
            sentiment_counts['POSITIVE'] += df.groupby('Decade')[sentiment_col].apply(lambda x: (x > 0).sum()).values
            sentiment_counts['NEGATIVE'] += df.groupby('Decade')[sentiment_col].apply(lambda x: (x < 0).sum()).values
        else:
            # Use positive and negative labels
            positive_counts = df[df[sentiment_col] == 'POSITIVE'].groupby('Decade').size()
            negative_counts = df[df[sentiment_col] == 'NEGATIVE'].groupby('Decade').size()
            sentiment_counts['POSITIVE'] += positive_counts.reindex(sentiment_counts.index, fill_value=0)
            sentiment_counts['NEGATIVE'] += negative_counts.reindex(sentiment_counts.index, fill_value=0)

    # Normalize the counts
    sentiment_counts['Total'] = sentiment_counts['POSITIVE'] + sentiment_counts['NEGATIVE']
    sentiment_counts['Positive_Percentage'] = (sentiment_counts['POSITIVE'] / sentiment_counts['Total']) * 100
    sentiment_counts['Negative_Percentage'] = (sentiment_counts['NEGATIVE'] / sentiment_counts['Total']) * 100
    sentiment_counts = sentiment_counts.sort_index(ascending=True)
    sentiment_normalized = sentiment_counts[['Positive_Percentage', 'Negative_Percentage']]

    # Create the stacked bar chart with Plotly
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=sentiment_counts.index,
        y=sentiment_counts['Positive_Percentage'],
        name='Positive sentiment',
        marker_color='blue'
    ))

    fig.add_trace(go.Bar(
        x=sentiment_counts.index,
        y=sentiment_counts['Negative_Percentage'],
        name='Negative sentiment',
        marker_color='orange'
    ))

    # Update layout
    fig.update_layout(
        title='Combined Sentiment Evolution by Decade (Across All Techniques)',
        xaxis=dict(title='Decade'),
        yaxis=dict(title='Percentage of Sentiment', ticksuffix='%'),
        barmode='stack',
        legend=dict(title='Sentiment'),
        template='plotly_white'
    )

    # Show the plot
    fig.show()
    

def plot_top_movie_genres_by_sentiment(df, x=5):
    """
    Plot the top x movie genres categorized by sentiment.

    Arguments:
        df: the DataFrame containing movie data.
        x: number of top genres to display.
    """
    # Define the sentiment columns
    sentiment_columns = {
#         'HuggingFace': 'HFT_Sentiment',   # categorical (POSITIVE, NEGATIVE)
        'TextBlob': 'TB_Sentiment',       # numerical (positive if > 0, negative if < 0)
        'VADER': 'VADER_Sentiment',       # numerical (positive if > 0, negative if < 0)
        'Emotions': 'Emotions_Sentiment', # categorical (POSITIVE, NEGATIVE)
        'NLTK': 'NLTK_Sentiment'          # categorical (POSITIVE, NEGATIVE)
    }

    # Convert numerical sentiment columns to POSITIVE/NEGATIVE labels
    df['TB_Sentiment_Label'] = df['TB_Sentiment'].apply(lambda x: 'POSITIVE' if x > 0 else 'NEGATIVE')
    df['VADER_Sentiment_Label'] = df['VADER_Sentiment'].apply(lambda x: 'POSITIVE' if x > 0 else 'NEGATIVE')
    
    # [HELPER] Create the "overall_sentiment" column
    def determine_overall_sentiment(row):
        # Gather all sentiment labels
        labels = [
    #             row['HFT_Sentiment'],
            row['TB_Sentiment_Label'],
            row['VADER_Sentiment_Label'],
            row['Emotions_Sentiment'],
            row['NLTK_Sentiment']
        ]
        # Count occurrences of POSITIVE and NEGATIVE
        positive_count = labels.count('POSITIVE')
        negative_count = labels.count('NEGATIVE')
        # Determine overall sentiment
        return 'POSITIVE' if positive_count > negative_count else 'NEGATIVE'

    df['Overall_sentiment'] = df.apply(determine_overall_sentiment, axis=1)

    # Separate the dataframe into subsets for negative and positive sentiments
    negative_subset = df[df['Overall_sentiment'] == 'NEGATIVE']
    positive_subset = df[df['Overall_sentiment'] == 'POSITIVE']
    
    # [HELPER] Plot pie chart for a subset
    def plot_pie(subset, title):
        if subset.empty:
            print(f"No data available for {title}.")
            return
        # Group by 'Grouped_genres' and count occurrences
        genre_counts = subset['Grouped_genres'].explode().value_counts().reset_index(name='Count')
        genre_counts.columns = ['Grouped_genres', 'Count']

        # Get top x genres
        top_x_genre = genre_counts.head(x)

        # Pie chart
        fig = px.pie(
            top_x_genre,
            values='Count',
            names='Grouped_genres',
            title=title,
            labels={
                'Grouped_genres': 'Genre',
                'Count': 'Count'
            },
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        # Customize layout
        fig.update_traces(textinfo='percent+label', pull=[0.1 if i == 0 else 0 for i in range(len(top_x_genre))])
        fig.update_layout(height=600, width=600)
        fig.show()

    plot_pie(negative_subset, f'Top {x} Movie Genres for Negative Sentiment')
    plot_pie(positive_subset, f'Top {x} Movie Genres for Positive Sentiment')

def plot_sunburst_genres_sentiment_emotions(df, theme, x=5):
    """
    Plots a sunburst chart with the hierarchy: Theme → Top Genres → Sentiment → Emotions.

    Arguments:
        df: the DataFrame containing movie data.
        theme: theme of the movies in the dataframe.
        x: number of top genres to include in the chart (default is 10).
    """

    (positive_emotions, negative_emotions) = emotions_group()

    df = df.copy()
    
    # Get the movie genres
    df['Movie_genres'] = df['Movie_genres'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    df_exploded = df.explode('Movie_genres')
    
    top_genres = (
        df_exploded['Movie_genres']
        .value_counts()
        .nlargest(x)
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
