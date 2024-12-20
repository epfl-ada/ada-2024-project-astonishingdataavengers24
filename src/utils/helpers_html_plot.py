import pandas as pd
from src.utils.helpers_sentiment_analysis import *
from src.utils.helpers_temporality_analysis import *
from src.utils.helpers_movies_genres import *

def plot_to_html(df, theme):
    """
    Creates html plots to put in the website.
    
    Arguments:
        - df: the DataFrame containing the movie data.
        - theme: theme of the movie (put everything in small caps).
    """
    themes = ['cold_war', 'economy', 'gender_equality', 'health', 'technology', 'vietnam', 'ww2']
    if theme not in themes:
        raise ValueError(f"Invalid theme. Choose from: {', '.join(themes)}. (Put everything in lowercase).")
        
    if theme == "cold_war":
        theme_plot = "Cold War"
    elif theme == "economy":
        theme_plot = "Economy"
    elif theme == "gender_equality":
        theme_plot = "Gender Equality"
    elif theme == "health":
        theme_plot = "Health"
    elif theme == "technology":
        theme_plot = "Technology"
    elif theme == "vietnam":
        theme_plot = "Vietnam War"
    elif theme == "ww2":
        theme_plot = "World War II"
        
    path_to_folder = f'../../figures/plots/{theme}/'
    
    fig = plot_movie_frequency(df, theme_plot, time_unit='Year')
    fig.write_html(path_to_folder + f'{theme}_movie_frequency_year.html')

    fig = plot_movie_frequency(df, theme_plot, time_unit='Decade')
    fig.write_html(path_to_folder + f'{theme}_movie_frequency_decade.html')

    fig = plot_overall_top_genres(df, theme_plot, x=10)
    fig.write_html(path_to_folder + f'{theme}_overall_top_genres.html')

    fig = plot_top_genres_by_decade(df, theme_plot, x=5)
    fig.write_html(path_to_folder + f'{theme}_top_genres_by_decade.html')

    fig = plot_emotion_counts(df, theme_plot)
    fig.write_html(path_to_folder + f'{theme}_emotion_counts.html')

    fig = plot_emotion_counts_by_decade(df, theme_plot)
    fig.write_html(path_to_folder + f'{theme}_emotion_counts_by_decade.html')

    fig = plot_emotion_sentiment_counts(df, theme_plot)
    fig.write_html(path_to_folder + f'{theme}_emotion_sentiment_counts.html')

    fig = plot_all_sentiments(df, theme_plot)
    fig.write_html(path_to_folder + f'{theme}_all_sentiments.html')

    fig = plot_combined_sentiment_by_decade(df, theme_plot)
    fig.write_html(path_to_folder + f'{theme}_combined_sentiment_by_decade.html')

    fig = plot_top_movie_genres_by_sentiment(df, theme_plot, x=5)
    fig['negative'].write_html(path_to_folder + f'{theme}_top_movie_genres_by_negative_sentiment.html')
    fig['positive'].write_html(path_to_folder + f'{theme}_top_movie_genres_by_positive_sentiment.html')

    fig = plot_sunburst_genres_sentiment_emotions(df, theme_plot)
    fig.write_html(path_to_folder + f'{theme}_sunburst_genres_sentiment_emotions.html')

    fig = plot_movies_and_news_frequency(df, theme_plot, time_unit='Year')
    fig.write_html(path_to_folder + f'{theme}_movies_and_news_frequency_year.html')

    fig = plot_movies_and_news_frequency(df, theme_plot, time_unit='Decade')
    fig.write_html(path_to_folder + f'{theme}_movies_and_news_frequency_decade.html')