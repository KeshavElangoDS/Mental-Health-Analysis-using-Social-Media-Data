"""
data_cleaning.py : Module for loading, processing, and cleaning mental health-related subreddit data.

This module contains functions for:
- Loading and combining multiple CSV files into a single DataFrame.
- Labeling subreddits as 'mental_health' or 'non_mental_health'.
- Cleaning text by removing non-ASCII characters and handling special HTML entities.

Functions:
- load_and_combine_dataset(data_folder): Loads and combines CSV files, extracting the first four columns.
- label_mental_health_subreddits(mental_health_df): Classifies subreddits as either related to mental health or not.
- clean_text(text): Cleans input text by removing non-ASCII characters and replacing HTML entities.
"""

import pandas as pd
from pathlib import Path
import os
import re


from pathlib import Path
import os
import pandas as pd

def load_and_combine_dataset(data_folder: str) -> pd.DataFrame:
    """
    Loads and combines CSV files from a specified folder, extracting only the first four columns from each file.

    Args:
        data_folder (str): The relative path to the folder containing the CSV files to be loaded.

    Returns:
        pd.DataFrame: A DataFrame containing the combined data from all CSV files, including only the first four columns.
        
    Notes:
        Assumes all CSV files in the provided folder are structured similarly and contain at least four columns.
        Returns an empty DataFrame if no CSV files are found.
    """
    current_working_folder = Path.cwd()
    current_folder_abs = os.path.abspath(current_working_folder)

    data_folder_path = os.path.join(current_folder_abs, data_folder)
    csv_files = [f for f in os.listdir(data_folder_path) if f.endswith('.csv')]

    data_frames = []

    for file in csv_files:
        file_path = os.path.join(data_folder_path, file)
        df = pd.read_csv(file_path)
        df_first_four_columns = df.iloc[:, :4]
        data_frames.append(df_first_four_columns)

    if not data_frames:
        return pd.DataFrame()

    mental_health_df = pd.concat(data_frames, ignore_index=True)
    return mental_health_df


def label_mental_health_subreddits(mental_health_df: pd.DataFrame):
    """
    Labels the rows in a DataFrame based on subreddit categories, classifying them as either mental health-related or non-mental health-related.

    Args:
        mental_health_df (pd.DataFrame): A DataFrame containing subreddit data, with a 'subreddit' column.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The input DataFrame with an additional 'target' column indicating mental health-related or not.
            - pd.DataFrame: A DataFrame with the subreddits and their corresponding classification ('mental_health' or 'non_mental_health').
        
    Notes:
        The 'subreddit' column in the input DataFrame is used to classify each subreddit. 
        Subreddits in the predefined mental health list will be labeled as 'mental_health', others will be labeled 'non_mental_health'.
    """

    data_subreddits = [
    'EDAnonymous', 'addiction', 'alcoholism', 'adhd', 'anxiety', 'autism',
    'bipolarreddit', 'bpd', 'depression', 'healthanxiety', 'lonely', 'ptsd',
    'schizophrenia', 'socialanxiety', 'suicidewatch', 'mentalhealth', 'COVID19_support',
    'conspiracy', 'divorce', 'fitness', 'guns', 'jokes', 'legaladvice', 'meditation',
    'parenting', 'personalfinance', 'relationships', 'teaching'
    ]

    mental_health_subreddits = [
    'EDAnonymous', 'addiction', 'alcoholism', 'adhd', 'anxiety', 'autism',
    'bipolarreddit', 'bpd', 'depression', 'healthanxiety', 'lonely', 'ptsd',
    'schizophrenia', 'socialanxiety', 'suicidewatch', 'mentalhealth', 'COVID19_support'
    ]

    subreddits_df = pd.DataFrame(data_subreddits, columns=['subreddit'])
    subreddits_df['class'] = subreddits_df['subreddit'].apply(
        lambda x: x if x in mental_health_subreddits else 'non_mental_health'
    )

    # Map the 'target' column in mental_health_df based on the subreddits_df class column
    mental_health_df['target'] = mental_health_df['subreddit'].str.strip().apply(
        lambda x: x if x in list(subreddits_df['subreddit']) else 'non_mental_health'
    )
    
    # Ensure the target column only has 'mental_health' or 'non_mental_health' values
    mental_health_df['target'] = mental_health_df['target'].apply(
        lambda x: subreddits_df[subreddits_df['subreddit'] == x]['class'].values[0] 
        if x in subreddits_df['subreddit'].values else 'non_mental_health'
    )

    return mental_health_df, subreddits_df

def clean_text(text):
    """
    Cleans the input text by removing non-ASCII characters, ensuring only valid characters remain.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text with non-ASCII characters removed and '&amp;' replaced with '&'.
        
    Notes:
        The function performs basic sanitization to remove any unwanted characters and special HTML entities.
    """
    cleaned_text = ''.join([char for char in text if ord(char) < 128])
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)
    cleaned_text = cleaned_text.replace("&amp;", "&")
    
    return cleaned_text

