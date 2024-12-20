import pandas as pd

def df_theme_year(theme, year=None, decade=None):
    """
    Filters dataframe with specific theme, and optionnally either by year or decade.
    
    Arguments:
        - theme: themes we want to find in the articles.
        - year: opt. year of the article.
        - decade: opt. decade of the article.
        
    Returns:
        A pd.DataFrame with only articles of a theme, and eventually from a specific year / decade.
    """
    
    themes = ["Cold War", "Economy", "Gender Equality", "Health", "Technology", "Vietnam War", "World War II"]
    if theme not in themes:
        raise ValueError(f"Invalid theme. Choose from: {', '.join(themes)}.")
        
    if theme == "Cold War":
        theme_column = "cold_war"
    elif theme == "Economy":
        theme_column = "economy"
    elif theme == "Gender Equality":
        theme_column = "gender_equality"
    elif theme == "Health":
        theme_column = "health"
    elif theme == "Technology":
        theme_column = "technology"
    elif theme == "Vietnam War":
        theme_column = "vietnam"
    elif theme == "World War II":
        theme_column = "ww2"
        
    df_news = pd.read_csv('../../data/df_news/cosine_similarity_news_cleaned.csv')
    
    df_news = df_news[df_news[theme_column] == 1]
    
    if year is not None:
        return df_news[df_news['year_x'] == year]
    elif decade is not None:
        return df_news[df_news['decade'] == decade]
    else:
        return df_news