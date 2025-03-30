"""
"""

import pandas as pd
from pathlib import Path
import os
import re


def load_and_combine_dataset(data_folder: str):
    """
    """

    current_working_folder = Path.cwd()
    current_folder_abs = os.path.abspath(current_working_folder)

    data_folder_path = os.path.join(current_folder_abs, data_folder)
    csv_files = [f for f in os.listdir(data_folder_path) if f.endswith('.csv')]

    data_frames = []

    #extract the first 4 columns
    for file in csv_files:
        file_path = os.path.join(data_folder_path, file)
        df = pd.read_csv(file_path)
        df_first_four_columns = df.iloc[:, :4]
        data_frames.append(df_first_four_columns)

    mental_health_df = pd.concat(data_frames, ignore_index=True)

    return mental_health_df

def label_mental_health_subreddits(mental_health_df: pd.DataFrame):
    """
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
    """
    cleaned_text = ''.join([char for char in text if ord(char) < 128])
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)
    cleaned_text = cleaned_text.replace("&amp;", "&")
    
    return cleaned_text

