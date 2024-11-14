import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_top_genres(top_x, genre_distribution):
    """
    Plot the top x genres per decade for each genre.

    Arguments:
        top_x: the number of top genres per decade
        genre_distribution: the DataFrame grouped by decade and genres
    """
    # Top x genres per decade
    top_x_genres_per_decade = genre_distribution.groupby(['Decade']).head(top_x)

    # Normalize the counts 
    decade_totals = top_x_genres_per_decade.groupby('Decade')['Count'].transform('sum')
    top_x_genres_per_decade = top_x_genres_per_decade.copy()  # to avoid warnings
    top_x_genres_per_decade['Percentage'] = (top_x_genres_per_decade['Count'] / decade_totals) * 100

    # Pivot the data for the stacked bar chart
    top_genres_normalized = top_x_genres_per_decade.pivot(index='Decade', columns='Grouped_genres', values='Percentage').fillna(0)

    # Color palette with color for each genre
    unique_genres = top_genres_normalized.columns
    colors = sns.color_palette("Spectral", n_colors=len(unique_genres)) 
    genre_colors = dict(zip(unique_genres, colors))

    # Plot the data as a stacked bar chart
    plt.figure(figsize=(14, 8))
    top_genres_normalized.plot(
        kind='bar',
        stacked=True,
        color=[genre_colors[genre] for genre in top_genres_normalized.columns], 
        width=0.8,
        ax=plt.gca()
    )

    plt.title(f'Top {top_x} Movie Genres By Decade')
    plt.xlabel('Decade')
    plt.ylabel('Percentage of Total Genres (%)')
    plt.legend(title='Genres', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid()
    plt.show()
    
def plot_top_genres_overall(x, all_genre_distr):
    """
    Plot a pie chart for the top x genres overall.

    Arguments:
        x: the number of top genres to display
        all_genre_distr: the DataFrame grouped by movie genre
    """
    # Get the top x genres overall
    top_x_genre_overall = all_genre_distr.head(x)

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
    
def plot_top_genres_by_continent(data, top_x=5):
    """
    Plot the top x genres by decade for each continent.

    Arguments:
        data: the DataFrame containing the data
        top_x: the number of top genres per decade to display
    """
    #Count Occurrences of grouped genres per decade per continent
    genre_counts = data.groupby(['Decade', 'Movie_continent', 'Grouped_genres']).size().reset_index(name='Count')

    # Get unique continents
    filtered_genre_counts = genre_counts[genre_counts['Movie_continent'] != 'Unknown']
    continents = filtered_genre_counts['Movie_continent'].unique()
    
   
    for continent in continents:
        continent_data = genre_counts[genre_counts['Movie_continent'] == continent]
        
        # Get top x genres per decade
        top_genres_per_decade = continent_data.groupby(['Decade', 'Grouped_genres'])['Count'].sum().reset_index()
        top_genres_per_decade = top_genres_per_decade.sort_values(['Decade', 'Count'], ascending=[True, False])
        top_genres_per_decade = top_genres_per_decade.groupby('Decade').head(top_x)

        # Normalize
        decade_totals = top_genres_per_decade.groupby('Decade')['Count'].transform('sum')
        top_genres_per_decade['Percentage'] = (top_genres_per_decade['Count'] / decade_totals) * 100

        # Pivot the data for a stacked bar chart
        top_genres_normalized = top_genres_per_decade.pivot(index='Decade',columns='Grouped_genres',values='Percentage').fillna(0)

        # Plot the data as a stacked bar chart 
        plt.figure(figsize=(14, 8))
        top_genres_normalized.plot(
            kind='bar', 
            stacked=True, 
            colormap="tab20", 
            width=0.8, 
            ax=plt.gca()
        )

        plt.title(f'Top {top_x} Movie Genres by Decade in {continent}')
        plt.xlabel('Decade')
        plt.ylabel('Percentage of Total Genres (%)')
        plt.legend(title='Genres', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid()
        plt.show()
    
def plot_top_genres_by_continent_overall(data, top_x=5):
    """
    Plot the evolution of the overall top x genres per continent 

    Arguments:
        data: DataFrame containing the data
        top_x: the number of top genres overall to display
    """
    #Identify the top x genres overall
    overall_genre_counts = data.groupby('Grouped_genres').size().reset_index(name='Count')
    top_genres = overall_genre_counts.nlargest(top_x, 'Count')['Grouped_genres'].tolist()

    # Filter the dataset to include only the top genres
    filtered_data = data[data['Grouped_genres'].isin(top_genres)]

    # Count occurences of genres per decade and per continent
    genre_evolution = filtered_data.groupby(['Decade', 'Movie_continent', 'Grouped_genres']).size().reset_index(name='Count')
    
    # Normalize the counts per decade within each continent
    genre_evolution['Total'] = genre_evolution.groupby(['Decade', 'Movie_continent'])['Count'].transform('sum')
    genre_evolution['Percentage'] = (genre_evolution['Count'] / genre_evolution['Total']) * 100

    # Drop 'Unknown' continent and get unique continents
    filtered_genre_evolution = genre_evolution[genre_evolution['Movie_continent'] != 'Unknown']
    continents = filtered_genre_evolution['Movie_continent'].unique()

    for continent in continents:
        # Filter data for the current continent
        continent_data = genre_evolution[genre_evolution['Movie_continent'] == continent]

        # Plot the data using a line plot
        plt.figure(figsize=(14, 8))
        sns.lineplot(
            data=continent_data,
            x='Decade',
            y='Percentage', 
            hue='Grouped_genres',  
            marker='o',
            palette='tab10'
        )
        plt.title(f'Evolution of Top {top_x} Movie Genres in {continent}')
        plt.xlabel('Decade')
        plt.ylabel('Percentage of Total Genres (%)')
        plt.legend(title='Genres', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.grid()
        plt.tight_layout()
        plt.show()
        
