import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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

def plot_articles_with_theme(theme, year=None, decade=None):
    """
    Plot articles by a given theme, and optionally by year or decade, and displays them in a table format.
    
    Arguments:
        - theme: theme we want to filter articles by.
        - year: opt. specific year of the article.
        - decade: opt. specific decade of the article.
        
    Returns:
        A Plotly table with the articles filtered by the specified theme, year, or decade.
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
    
    # Filter the dataframe using the given theme, year, or decade
    df_filtered = df_theme_year(theme, year, decade)
    
    # Ensure the dataframe contains 'title', 'year_x', 'decade', and 'excerpt' columns
    df_filtered['title_truncated'] = df_filtered['title'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)
    
    # Create the table
    table_data = go.Figure(data=[go.Table(
        header=dict(values=["Year", "Decade", "Title", "Excerpt"]),
        cells=dict(values=[
            df_filtered['year_x'],
            df_filtered['decade'], 
            df_filtered['title_truncated'], 
            df_filtered['excerpt']  
        ]),
        hoverinfo="x+text", 
        customdata=df_filtered[['year_x', 'decade', 'title', 'excerpt']].values
    )])

    # Customize layout
    table_data.update_layout(
        title=f"Articles about {theme.capitalize()}",
        showlegend=False
    )
    
    if year is not None:
        table_data.update_layout(
            title=f"Articles about {theme.capitalize()} in {year}",
            showlegend=False
        )
    elif decade is not None:
        table_data.update_layout(
            title=f"Articles about {theme.capitalize()} in the {decade}s",
            showlegend=False
        )
    
    return table_data
