import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

COLOR_PALETTE = px.colors.qualitative.Prism
POSITIVE_MARKER = px.colors.qualitative.Prism[2]  # Cyan
NEGATIVE_MARKER = px.colors.qualitative.Prism[7]   # Red

def preprocess_text(text):
    """Preprocess text: tokenization, lemmatization"""
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

"""

Sentiment Analysis

"""

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

def perform_sentiment_analysis(df):
    """
    Perform various sentiment and emotion analyses on the on the plot summary.

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
    df.loc[:, 'Plot_summary'] = df['Plot_summary'].apply(preprocess_text)

    # NLTK Sentiment Analysis
    df.loc[:, 'NLTK_Sentiment'] = df['Plot_summary'].apply(get_nltk_sentiment)

    # TextBlob Sentiment Analysis
    df.loc[:, 'TB_Sentiment'] = df['Plot_summary'].apply(get_textblob_sentiment)

    # VADER Sentiment Analysis
    df.loc[:, 'VADER_Sentiment'] = df['Plot_summary'].apply(get_vader_sentiment)

    # Emotion Analysis with NRC Lexicon
    df.loc[:, 'Emotions'] = df['Plot_summary'].apply(extract_emotions)

    # Sentiment from Emotions
    df.loc[:, 'Emotions_Sentiment'] = df['Emotions'].apply(classify_sentiment_from_emotions)

    return df

def count_sentiments(df):
    """
    Count the number of positive and negative sentiments in each sentiment analysis column.

    Arguments:
        df: the DataFrame containing movie data.

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

def plot_sentiment_by_decade(df, theme, technique='NLTK'):
    """
    Plot sentiment evolution by decade for a specific technique.
    
    Arguments:
        df: DataFrame containing the data.
        theme: Theme to analyze.
    """
    # Ensure given technique is valid
    techniques = ['NLTK', 'TextBlob', 'VADER', 'Emotions']
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

    if technique in ['TextBlob', 'VADER']:
        # Use numerical sentiment values for TextBlob and VADER
        sentiment_counts = df.groupby('Decade')[sentiment_col].apply(lambda x: (x > 0).sum()).reset_index(name='POSITIVE')
        sentiment_counts['NEGATIVE'] = df.groupby('Decade')[sentiment_col].apply(lambda x: (x < 0).sum()).reset_index(name='NEGATIVE')['NEGATIVE']
    else:
        # Use positive and negative labels for NLTK and Emotions
        sentiment_counts = df.groupby('Decade')[sentiment_col].value_counts().unstack(fill_value=0).reset_index()
        sentiment_counts.rename(columns={positive_label: 'POSITIVE', negative_label: 'NEGATIVE'}, inplace=True)

    # Normalize counts
    sentiment_counts['Total'] = sentiment_counts['POSITIVE'] + sentiment_counts['NEGATIVE']
    sentiment_counts['Positive_Percentage'] = (sentiment_counts['POSITIVE'] / sentiment_counts['Total']) * 100
    sentiment_counts['Negative_Percentage'] = (sentiment_counts['NEGATIVE'] / sentiment_counts['Total']) * 100

    # Create traces for the bar chart
    positive_trace = go.Bar(
        x=sentiment_counts['Decade'],
        y=sentiment_counts['Positive_Percentage'],
        name='Positive',
        marker_color=POSITIVE_MARKER  # Use Prism Cyan
    )

    negative_trace = go.Bar(
        x=sentiment_counts['Decade'],
        y=sentiment_counts['Negative_Percentage'],
        name='Negative',
        marker_color=NEGATIVE_MARKER  # Use Prism Red
    )
    
    return positive_trace, negative_trace


def plot_all_sentiments(df, theme):
    """
    Plot sentiment evolution for all techniques in one figure with four subplots.
    
    Arguments:
        df: DataFrame containing the data.
        theme: Theme to analyze.
        
    Returns:
        Plotly figure.
    """
    techniques = ['NLTK', 'TextBlob', 'VADER', 'Emotions']
    
    # Create a 2x2 grid of subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"{technique} Sentiment" for technique in techniques],
        shared_xaxes=True, shared_yaxes=True
    )

    # Add plots for each technique
    for i, technique in enumerate(techniques):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        # Get positive and negative traces
        positive_trace, negative_trace = plot_sentiment_by_decade(df, theme, technique)
        
        # Group traces by legend group to ensure toggling works globally
        positive_trace.update(legendgroup='Positive', showlegend=(i == 0))
        negative_trace.update(legendgroup='Negative', showlegend=(i == 0))
        
        # Add traces to the subplot
        fig.add_trace(positive_trace, row=row, col=col)
        fig.add_trace(negative_trace, row=row, col=col)

    fig.update_layout(
        title=f'Sentiment Evolution by Decade for Theme: {theme}',
        xaxis_title='Decade',
        yaxis_title='Percentage of Sentiment',
        barmode='stack',
        legend_title='Sentiment',
        height=800,
        width=1000,
        template='plotly_white',
    )
    
    return fig


def plot_combined_sentiment_by_decade(df, theme):
    """
    Plot combined sentiment evolution (positive or negative) over decades by aggregating sentiment across all techniques.
    
    Arguments:
        df: the DataFrame containing movie data.
        theme: String representing the theme.
        
    Returns:
        Plotly figure.
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

    # Stacked bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=sentiment_counts.index,
        y=sentiment_counts['Positive_Percentage'],
        name='Positive',
        marker_color=POSITIVE_MARKER
    ))

    fig.add_trace(go.Bar(
        x=sentiment_counts.index,
        y=sentiment_counts['Negative_Percentage'],
        name='Negative',
        marker_color=NEGATIVE_MARKER
    ))

    fig.update_layout(
        title=f'Sentiment Evolution across all Techniques by Decade for {theme} Theme',
        xaxis=dict(title='Decade'),
        yaxis=dict(title='Percentage of Sentiment', ticksuffix='%'),
        barmode='stack',
        legend=dict(title='Sentiment')
    )

    return fig
    
def plot_sentiment_pie_charts(df, theme, x=5):
    """
    Plot interactive pie charts of the top x genres for movies categorized by overall sentiment.

    Arguments:
        df: the DataFrame containing movie data.
        theme: String representing the theme.
        x: Number of top genres to display.
        
    Returns:
        Plotly figure.
    """
    # Define the sentiment columns
    sentiment_columns = {
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
        labels = [
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

        # Get top x genres
        top_x_genre = genre_counts.head(x)

        # Pie chart
        fig = px.pie(
            top_x_genre,
            values='Count',
            names='Grouped_genres',
            title=title,
            color_discrete_sequence=COLOR_PALETTE
        )

        # Customize layout
        fig.update_traces(textinfo='percent+label', pull=[0.1 if i == 0 else 0 for i in range(len(top_x_genre))])
        fig.update_layout(height=600, width=600)

    plot_pie(negative_subset, f'Top {x} Movie Genres for Negative Sentiment for {theme} Theme')
    plot_pie(positive_subset, f'Top {x} Movie Genres for Positive Sentiment for {theme} Theme')
    
    return fig
    
    
def plot_top_movie_genres_by_sentiment(df, theme, x=5):
    """
    Plot the top x movie genres categorized by sentiment.
    
    Arguments:
        df: the DataFrame containing movie data.
        theme: String representing the theme.
        x: number of top genres to display.
        
    Returns:
        A dictionary with figures for 'negative' and 'positive' sentiments.
    """
    # Define the sentiment columns
    sentiment_columns = {
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
            return None
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
            color_discrete_sequence=COLOR_PALETTE  # Default color palette
        )
        # Customize layout
        fig.update_traces(textinfo='percent+label', pull=[0.1 if i == 0 else 0 for i in range(len(top_x_genre))])
        fig.update_layout(height=600, width=600)
        return fig

    # Generate and return the figures
    negative_fig = plot_pie(negative_subset, f'Top {x} Movie Genres for Negative Sentiment for {theme} Theme')
    positive_fig = plot_pie(positive_subset, f'Top {x} Movie Genres for Positive Sentiment for {theme} Theme')
    
    return {
        'negative': negative_fig,
        'positive': positive_fig
    }

"""

Emotions Analysis

"""
    
def create_emotions_column(df):
    """
    Creates emotions column in the given dataframe.
    
    Arguments:
        df: the DataFrame containing movie data.
    """
    df = df.copy()

    # Apply VADER Sentiment Analysis
    df.loc[:, 'VADER_Sentiment'] = df['Plot_summary'].apply(get_vader_sentiment)

    # Apply Emotion Analysis
    df.loc[:, 'Emotions'] = df['Plot_summary'].apply(extract_emotions)

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
        
    Returns:
        Plotly figure.
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
        title=f"Total Counts of Emotions Across Movies for {theme} Theme",
        labels={'Emotion': 'Emotion', 'Count': 'Count'},
        text='Count', 
        color_discrete_sequence=COLOR_PALETTE
    )
    
    fig.update_layout(xaxis_title="Emotion", yaxis_title="Total Count")
    
    return fig
    
def plot_emotion_counts_by_decade(df, theme):
    """
    Plot the count of emotions across movies by decade.
    
    Arguments:
        df: the DataFrame containing movie data.
        theme: theme of the movies in the dataframe.
        
    Returns:
        Plotly figure.
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

    # Calculate total counts per decade
    total_counts_by_decade = emotions_summary.groupby('Decade')['Count'].transform('sum')

    # Normalize emotion counts by total counts per decade
    emotions_summary['Normalized_Count'] = emotions_summary['Count'] / total_counts_by_decade

    # Line chart
    fig = px.line(
        emotions_summary,
        x='Decade',
        y='Normalized_Count',
        color='Emotion',
        title=f"Counts of Emotions by Decade for {theme} Theme",
        labels={'Decade': 'Decade', 'Count': 'Count', 'Emotion': 'Emotion'},
        markers=True,
        color_discrete_sequence=COLOR_PALETTE
    )
    fig.update_layout(
        xaxis_title="Decade",
        yaxis_title="Total Emotion Count",
        xaxis=dict(dtick=10)
    )
    
    return fig
    
def plot_emotion_sentiment_counts(df, theme):
    """
    Plot a bar chart showing the count of emotions categorized as sentiments (positive or negative).
    
    Arguments:
        df: the DataFrame containing movie data.
        theme: theme of the movies in the dataframe.
        
    Returns:
        Plotly figure.
    """
    # Define emotion groups
    positive_emotions, negative_emotions = emotions_group()

    # Ensure the genres column is parsed correctly
    df = df.copy()
    

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

    # Bar chart
    fig = px.bar(
        sentiment_counts,
        x='Sentiment',
        y='Emotion_Count',
        color='Sentiment',
        title=f"Emotion Counts Categorized by Sentiment for Theme: {theme}",
        labels={'Sentiment': 'Sentiment', 'Emotion_Count': 'Total Emotion Count'},
        text='Emotion_Count',
        color_discrete_sequence=[NEGATIVE_MARKER, POSITIVE_MARKER]
    )

    fig.update_layout(xaxis_title="Sentiment", yaxis_title="Total Emotion Count")

    return fig
    
def plot_sunburst_genres_sentiment_emotions(df, theme, x=5):
    """
    Plot a sunburst chart with the hierarchy: Theme → Top Genres → Sentiment → Emotions.
    
    Arguments:
        df: the DataFrame containing movie data.
        theme: theme of the movies in the dataframe.
        x: number of top genres to include in the chart (default is 10).
        
    Returns:
        Plotly figure.
    """

    (positive_emotions, negative_emotions) = emotions_group()

    df = df.copy()

    # Get the movie genres
    df['Grouped_genres'] = df['Grouped_genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    df_exploded = df.explode('Grouped_genres')

    top_genres = (
        df_exploded['Grouped_genres']
        .value_counts()
        .nlargest(x)
        .index
        .tolist()
    )

    # Filter for rows with the top genres 
    df_exploded = df_exploded[df_exploded['Grouped_genres'].isin(top_genres)]

    # Expand the dictionary into individual rows
    emotion_data = []
    for _, row in df_exploded.iterrows():
        if isinstance(row['Emotions'], dict):
            for emotion, count in row['Emotions'].items():
                if row['Emotions_Sentiment'] == 'POSITIVE' and emotion in positive_emotions:
                    emotion_data.append({
                        'Theme': theme,
                        'Genre': row['Grouped_genres'],
                        'Sentiment': row['Emotions_Sentiment'],
                        'Emotion': emotion,
                        'Emotion_Count': count
                    })
                elif row['Emotions_Sentiment'] == 'NEGATIVE' and emotion in negative_emotions:
                    emotion_data.append({
                        'Theme': theme,
                        'Genre': row['Grouped_genres'],
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
        title=f"Overall chart for {theme} Theme",
        labels={
            'Theme': 'Theme',
            'Genre': 'Genre',
            'Sentiment': 'Sentiment',
            'Emotion': 'Emotion',
            'Emotion_Count': 'Emotion Count'
        },
        color='Emotion_Count', 
        color_continuous_scale='RdBu',
    )

    return fig
