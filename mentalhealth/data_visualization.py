"""
"""

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
from wordcloud import STOPWORDS

def plot_post_counts_by_subreddit(data, column='subreddit', figsize=(15, 8)):
    """
    Plots the post counts across subreddits.

    Parameters:
    data (pd.DataFrame): The dataframe containing the data.
    column (str): The column name containing subreddit information (default is 'subreddit').
    figsize (tuple): The figure size (default is (15, 8)).

    Returns:
    None
    """
    plt.figure(figsize=figsize)
    sns.countplot(data=data, y=column)
    plt.title(f'Post counts across {column}s')
    plt.ylabel(f'{column.capitalize()}s')
    plt.xlabel('Count')
    plt.show()

def plot_post_lengths_by_target(data, x_column='post_length', y_column='target', figsize=(15, 6), palette='viridis'):
    """
    Plots a barplot of post lengths against a target variable, grouped by target.

    Parameters:
    data (pd.DataFrame): The dataframe containing the data.
    x_column (str): The column name representing post length (default is 'post_length').
    y_column (str): The column name representing the target (default is 'target').
    figsize (tuple): The figure size (default is (15, 6)).
    palette (str): The color palette for the plot (default is 'viridis').

    Returns:
    None
    """
    plt.figure(figsize=figsize)
    sns.barplot(x=x_column, y=y_column, hue=y_column, data=data, palette=palette)
    plt.title(f'Post Lengths for Each {y_column.capitalize()}')
    plt.xlabel('Post Length')
    plt.ylabel(f'Reddit post ({y_column.capitalize()})')
    plt.show()

def generate_wordcloud(dataframe, text_column, width=800, height=400, background_color='white'):
    """
    Generates and displays a word cloud from the specified text column in a DataFrame.
    
    Parameters:
    - dataframe: pandas DataFrame containing the text data.
    - text_column: str, the column in the DataFrame that contains the text data to generate the word cloud.
    - width: int, the width of the word cloud image (default is 800).
    - height: int, the height of the word cloud image (default is 400).
    - background_color: str, background color of the word cloud (default is 'white').
    
    Returns:
    - A word cloud image displayed using matplotlib.
    """
    combined_text = " ".join(dataframe[text_column])
    
    wordcloud = WordCloud(
        width=width, 
        height=height, 
        background_color=background_color, 
        stopwords=STOPWORDS
    ).generate(combined_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Turn off the axes
    plt.show()

