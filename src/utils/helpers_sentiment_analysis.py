import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from transformers import pipeline
from nrclex import NRCLex

# nltk.download('vader_lexicon')
# nltk.download('punkt')
# nltk.download('wordnet')

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
    positive_emotions = ['positive', 'anticipation', 'surprise', 'joy', 'trust']
    negative_emotions = ['negative', 'anger', 'fear', 'disgust', 'sadness']
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

    Args:
        df (pd.DataFrame): Input dataframe on which we want to perform sentiment analysis.

    Returns:
        pd.DataFrame: Updated dataframe with new sentiment analysis columns.
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

def count_sentiments(df):
    """
    Count the number of positive and negative sentiments in each sentiment analysis column.

    Args:
        df (pd.DataFrame): Input dataframe containing sentiment analysis columns.

    Returns:
        dict: A dictionary with counts of positive and negative sentiments for each column.
    """
    counts = {
#         'HFT_Sentiment': {'Positive': 0, 'Negative': 0},
        'NLTK_Sentiment': {'Positive': 0, 'Negative': 0},
        'TB_Sentiment': {'Positive': 0, 'Negative': 0},
        'VADER_Sentiment': {'Positive': 0, 'Negative': 0},
        'Emotions_Sentiment': {'Positive': 0, 'Negative': 0}
    }
    # Count for NLTK Sentiment
    counts['NLTK_Sentiment']['Positive'] = (df['NLTK_Sentiment'] == 'POSITIVE').sum()
    counts['NLTK_Sentiment']['Negative'] = (df['NLTK_Sentiment'] == 'NEGATIVE').sum()

    # Count for TextBlob Sentiment
    counts['TB_Sentiment']['Positive'] = (df['TB_Sentiment'] > 0).sum()
    counts['TB_Sentiment']['Negative'] = (df['TB_Sentiment'] < 0).sum()

    # Count for VADER Sentiment
    counts['VADER_Sentiment']['Positive'] = (df['VADER_Sentiment'] > 0).sum()
    counts['VADER_Sentiment']['Negative'] = (df['VADER_Sentiment'] < 0).sum()

    # Count for Emotions Sentiment
    counts['Emotions_Sentiment']['Positive'] = (df['Emotions_Sentiment'] == 'POSITIVE').sum()
    counts['Emotions_Sentiment']['Negative'] = (df['Emotions_Sentiment'] == 'NEGATIVE').sum()
    
    # Count for Huggingface Sentiment
#     counts['HFT_Sentiment']['Positive'] = (df['HFT_Score'] > 0).sum()
#     counts['HFT_Sentiment']['Negative'] = (df['HFT_Score'] < 0).sum()

    # Format output as table
    result = "                   | Positive | Negative"
    result += "\n-------------------|----------|---------"
    for column, sentiment_counts in counts.items():
        result += f"\n{column:<18} | {sentiment_counts['Positive']:<8} | {sentiment_counts['Negative']:<7}"

    return result

def plot_movie_frequency_by_decade(df):
    """
    Plot the evolution of the frequency of movies per decade.

    Arguments:
        data: DataFrame containing the movie data with a 'Decade' column
    """
    # Count occurrences of movies per decade
    movie_evolution = df.groupby('Decade').size().reset_index(name='Count')
    
    plt.figure(figsize=(14, 8))
    sns.lineplot(
        data=movie_evolution,
        x='Decade',
        y='Count', 
        marker='o',
        color='green'
    )
    
    plt.title('Evolution of Movie Frequency by Decade')
    plt.xlabel('Decade')
    plt.ylabel('Number of Movies')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_sentiment_by_decade(df, technique='NLTK'):
    """
    Plot Sentiment Evolution (Positive/Negative) over Decades for a specified sentiment analysis technique.
    
    Args:
        df (pd.DataFrame): Input dataframe containing sentiment analysis columns.
        technique (str): Sentiment analysis technique to plot ('NLTK', 'TextBlob', 'VADER', 'Emotions').
    """
    # Ensure given technique valid
    valid_techniques = ['NLTK', 'TextBlob', 'VADER', 'Emotions']
    if technique not in valid_techniques:
        raise ValueError(f"Invalid technique. Choose from: {', '.join(valid_techniques)}")

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

    if technique in ['TextBlob', 'VADER']:
        # Use numerical sentiment values for textblob and vader
        sentiment_counts = df.groupby('Decade')[sentiment_col].apply(lambda x: (x > 0).sum()).reset_index(name='POSITIVE')
        sentiment_counts['NEGATIVE'] = df.groupby('Decade')[sentiment_col].apply(lambda x: (x < 0).sum()).reset_index(name='NEGATIVE')['NEGATIVE']
        sentiment_counts['POSITIVE'] = df.groupby('Decade')[sentiment_col].apply(lambda x: (x > 0).sum()).reset_index(name='POSITIVE')['POSITIVE']
    else:
        # Use positive and negative labels for nltk and emotions
        sentiment_counts = df.groupby('Decade')[sentiment_col].value_counts().unstack(fill_value=0)
        

    # Normalize counts 
    sentiment_counts['Total'] = sentiment_counts['POSITIVE'] + sentiment_counts['NEGATIVE']
    sentiment_counts['Positive_Percentage'] = (sentiment_counts['POSITIVE'] / sentiment_counts['Total']) * 100
    sentiment_counts['Negative_Percentage'] = (sentiment_counts['NEGATIVE'] / sentiment_counts['Total']) * 100
    sentiment_normalized = sentiment_counts[['Positive_Percentage', 'Negative_Percentage']]

    # Plot stacked bar chart
    plt.figure(figsize=(14, 8))
    sentiment_normalized.plot(
        kind='bar',
        stacked=True,
        color=['lightgreen', 'pink'],
        width=0.8,
        ax=plt.gca()
    )

    # Format plot
    plt.title(f'{technique} Sentiment Evolution by Decade')
    plt.xlabel('Decade')
    plt.ylabel('Percentage of Sentiment')
    plt.xticks(rotation=45)
    plt.legend(['Positive', 'Negative'], loc='upper left')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def plot_combined_sentiment_by_decade(df):
    """
    Plot combined Sentiment Evolution (Positive/Negative) over Decades by aggregating sentiment across all techniques.
    
    Args:
        df (pd.DataFrame): Input dataframe containing sentiment analysis columns.
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

    # Plot stacked bar chart
    plt.figure(figsize=(14, 8))
    sentiment_normalized.plot(
        kind='bar',
        stacked=True,
        color=['lightgreen', 'pink'],
        width=0.8,
        ax=plt.gca()
    )

    # Format plot
    plt.title(f'Combined Sentiment Evolution by Decade (Across All Techniques)')
    plt.xlabel('Decade')
    plt.ylabel('Percentage of Sentiment')
    plt.xticks(rotation=45)
    plt.legend(['Positive', 'Negative'], loc='upper left')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    
# small modification from movies_genres_utils
def plot_top_genres(top_x, df):
    """
    Plot the top x genres per decade for each genre.

    Arguments:
        top_x: the number of top genres per decade
        df: the DataFrame containing movie data with 'Grouped_genres' and 'Decade' columns
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
    top_x_genres_per_decade = genre_distribution.groupby(['Decade']).head(top_x)
    
    # Avoid warning using copy
    top_x_genres_per_decade = top_x_genres_per_decade.copy()
    
    # Normalize the counts by calculating the percentage for each genre
    decade_totals = top_x_genres_per_decade.groupby('Decade')['Count'].transform('sum')
    top_x_genres_per_decade['Percentage'] = (top_x_genres_per_decade['Count'] / decade_totals) * 100
    top_genres_normalized = top_x_genres_per_decade.pivot(index='Decade', columns='Grouped_genres', values='Percentage').fillna(0)

    # Color palette
    unique_genres = top_genres_normalized.columns
    colors = sns.color_palette("Spectral", n_colors=len(unique_genres)) 
    genre_colors = dict(zip(unique_genres, colors))

    # Stacked bar chart
    plt.figure(figsize=(14, 8))
    top_genres_normalized.plot(
        kind='bar',
        stacked=True,
        color=[genre_colors[genre] for genre in top_genres_normalized.columns], 
        width=0.8,
        ax=plt.gca()
    )

    # Format plot
    plt.title(f'Top {top_x} Movie Genres By Decade')
    plt.xlabel('Decade')
    plt.ylabel('Percentage of Total Genres (%)')
    plt.legend(title='Genres', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    
def plot_top_genres_overall(x, df):
    """
    Plot a pie chart for the top x genres overall.
    
    Arguments:
        x: the number of top genres to display
        all_genre_distr: the DataFrame containing movie genre information
    """
    # Group by 'Grouped_genres' and count occurrences
    df['Grouped_genres'] = df['Grouped_genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Exploding the 'Grouped_genres' column so that each genre becomes a separate row
    df_expanded = df.explode('Grouped_genres')
    genre_counts = df_expanded.groupby(['Grouped_genres']).size().reset_index(name="Count")
    genre_counts = genre_counts.sort_values(by=['Count'], ascending=False)

    # Get the top x genres overall
    top_x_genre_overall = genre_counts.head(x)

    # Plot with pie chart
    plt.figure(figsize=(10, 10))
    plt.pie(
        top_x_genre_overall['Count'], 
        labels=top_x_genre_overall['Grouped_genres'], 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=sns.color_palette("tab20", len(top_x_genre_overall))
    )

    # Title and display
    plt.title(f'Top {x} Movie Genres Overall')
    plt.show()
    
def plot_sentiment_pie_charts(df, x=5):
    """
    Plot pie charts of the top x genres for movies categorized by overall sentiment.

    Arguments:
        df: DataFrame containing sentiment analysis columns for each technique and genre information.
        x: Number of top genres to display.
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

    # Create the "overall_sentiment" column
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

    # Helper function to plot the pie chart for a subset
    def plot_pie(subset, title):
        if subset.empty:
            print(f"No data available for {title}.")
            return
        # Group by 'Grouped_genres' and count occurrences
        genre_counts = subset['Grouped_genres'].explode().value_counts().reset_index(name='Count')
        genre_counts.columns = ['Grouped_genres', 'Count']

        # Get the top x genres
        top_x_genre = genre_counts.head(x)

        # Plot with pie chart
        plt.figure(figsize=(10, 10))
        plt.pie(
            top_x_genre['Count'], 
            labels=top_x_genre['Grouped_genres'], 
            autopct='%1.1f%%', 
            startangle=140, 
            colors=sns.color_palette("tab20", len(top_x_genre))
        )
        plt.title(title)
        plt.show()

    # Plot for negative sentiment
    plot_pie(negative_subset, f'Top {x} Movie Genres for Negative Sentiment')

    # Plot for positive sentiment
    plot_pie(positive_subset, f'Top {x} Movie Genres for Positive Sentiment')
