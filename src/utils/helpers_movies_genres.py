import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import ast

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
        
