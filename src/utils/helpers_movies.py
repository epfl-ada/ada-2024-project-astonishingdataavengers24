import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import ast

COLOR_PALETTE =  px.colors.qualitative.Prism

def plot_main_themes_bubble_chart(themes, weights, period=None, genres=None):
    """
    Create a bubble chart to represent the main themes in a given period and genre.

    Arguments:
        themes: List of strings representing the themes.
        weights: List of integers representing the weights of the themes.
        period: String representing the period.
        genres: String representing the genre.
    """
    fig = go.Figure()

    # Add a scatter plot with bubble sizes corresponding to weights
    fig.add_trace(go.Scatter(
        x=[0, 1, 2, 3],  # Spread along x-axis for clarity
        y=[0, 0, 0, 0],  # Keep on a single line
        mode='markers+text',
        marker=dict(
            size=[w * 30 for w in weights],  # Scale marker size by weights
            color=['lightblue', 'pink', 'lightgreen', 'gold'],  #Distinct color 
            opacity=0.6
        ),
        text=themes,
        textposition="middle center",
        hoverinfo='text'
    ))

    title = f"Main Themes" + (f" in {period}" if period is not None else "") + (f" for {genres} Genre" if genres is not None else "")

    
    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400
    )

    fig.show()

def plot_movie_countries_world_map(df, theme, period=None):
    """
    Create a world map visualization of movie counts by country for a given theme.

    Arguments:
        df: the DataFrame containing movie data.
        theme: String representing the theme.
        period: String representing the decade.
        
    Returns:
        Plotly figure.
    """l
    df_copy = df.copy()

    # Parse 'Movie_countries' column to ensure it contains lists
    df_copy['Movie_countries'] = df_copy['Movie_countries'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

   
    df_expanded = df_copy.explode('Movie_countries')

  
    country_counts = df_expanded.groupby(['Movie_countries']).size().reset_index(name="Count")
    country_counts.columns = ['Country', 'Movie_Count']

    title = f"Movies by Country for Theme: {theme}" if period is None else f"Movies by Country for Theme: {theme} ({period})"

    # World map 
    fig = px.choropleth(
        country_counts,
        locations='Country',
        locationmode='country names',  #
        color='Movie_Count',
        hover_name='Country',
        color_continuous_scale='Viridis',
        title=title,
    )

    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        ),
        coloraxis_colorbar=dict(title="Number of Movies"),
    )

    return fig

def plot_overall_top_countries(df, theme, x=10):
    """
    Plot a pie chart for the top x movie-producing countries overall.
    
    Arguments:
        df: the DataFrame containing movie data.
        theme: String representing the theme.
        x: the number of top countries to display.
        
    Returns:
        Plotly figure.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Parse 'Movie_countries' column to ensure it contains lists
    df_copy['Movie_countries'] = df_copy['Movie_countries'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # Exploding the 'Movie_countries' column so that each country becomes a separate row
    df_expanded = df_copy.explode('Movie_countries')

    # Group by 'Movie_countries' and count occurrences
    country_counts = df_expanded.groupby(['Movie_countries']).size().reset_index(name="Count")
    country_counts = country_counts.sort_values(by=['Count'], ascending=False)

    # Get the overall top x countries
    top_x_countries = country_counts.head(x)

    # Pie chart
    fig = px.pie(
        top_x_countries, 
        values='Count', 
        names='Movie_countries',
        title=f'Top {x} Movie Countries for {theme} Theme',
        labels={
            'Movie_countries': 'Country',
            'Count': 'Count'
        },
        color_discrete_sequence=COLOR_PALETTE
    )

    fig.update_traces(textinfo='percent+label', pull=[0.1 if i == 0 else 0 for i in range(x)])
    fig.update_layout(height=600, width=600)

    return fig