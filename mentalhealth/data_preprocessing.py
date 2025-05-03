"""
data_preprocessing.py : Module for performing text preprocessing tasks on a given dataframe.

This module includes functions for calculating text statistics, tokenizing and lemmatizing text, 
and saving the processed data to a Parquet file. It also supports parallel processing to improve 
efficiency when working with large datasets.

Functions:
    - text_statistics: Calculate various text statistics (e.g., word count, unique words) for a given text column.
    - process_chunk: Process a chunk of the dataframe to compute text statistics.
    - parallel_text_statistics: Perform parallelized text statistics calculation across chunks of the dataframe.
    - process_and_lemmatize_chunk: Tokenize and lemmatize text data in a chunk of the dataframe.
    - tokenize_and_parallelize: Tokenize and lemmatize training data, then save the result to a Parquet file.
    - save_to_parquet_file: Save a dataframe to a Parquet file.
"""

import spacy
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import math

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

spacy_model = spacy.load('en_core_web_sm')

def text_statistics(df, text_column):
    """
    Calculates various text statistics for a given text column in a dataframe.

    Args:
        df (pandas.DataFrame): The dataframe containing the text data.
        text_column (str): The name of the column containing the text.

    Returns:
        dict: A dictionary containing the following statistics:
            - "total_words" (int): Total number of words in the text column.
            - "unique_words" (set): Set of unique words in the text column.
            - "total_sentences" (int): Total number of sentences in the text column.
            - "total_words_without_stopwords" (int): Total number of words excluding stopwords.
    """
    total_words = 0
    total_sentences = 0
    unique_words = set()
    total_words_without_stopwords = 0

    for text in df[text_column]:
        doc = spacy_model(str(text))

        total_sentences += len(list(doc.sents))
        
        # Tokenize words and count total words, unique words, and words without stopwords
        for token in doc:
            if token.is_alpha:  # Only consider alphabetic tokens (words)
                total_words += 1
                unique_words.add(token.text.lower())
                
                if not token.is_stop:
                    total_words_without_stopwords += 1

    stats = {
        "total_words": total_words,
        "unique_words": unique_words,
        "total_sentences": total_sentences,
        "total_words_without_stopwords": total_words_without_stopwords
    }
    return stats

# Function to run text_statistics on each chunk of the dataframe
def process_chunk(chunk, text_column):
    """
    Processes a chunk of the dataframe to calculate text statistics.

    Args:
        chunk (pandas.DataFrame): A chunk of the dataframe to process.
        text_column (str): The name of the column containing the text.

    Returns:
        dict: A dictionary containing the text statistics for the chunk.
    """
    return text_statistics(chunk, text_column)

def parallel_text_statistics(df, text_column, num_cores=3):
    """
    Calculates text statistics in parallel by splitting the dataframe into chunks.

    Args:
        df (pandas.DataFrame): The dataframe containing the text data.
        text_column (str): The name of the column containing the text.
        num_cores (int, optional): The number of CPU cores to use for parallel processing. Defaults to 3.

    Returns:
        dict: A dictionary containing the combined statistics across all chunks:
            - "total_words" (int): Total number of words in the text column.
            - "unique_words_count" (int): Count of unique words in the text column.
            - "total_sentences" (int): Total number of sentences in the text column.
            - "total_words_without_stopwords" (int): Total number of words excluding stopwords.
    """
    # Split the dataframe into chunks for parallel processing
    chunk_size = math.ceil(len(df) / num_cores)
    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    # Using ProcessPoolExecutor to run the function in parallel
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(executor.map(process_chunk, chunks, [text_column] * len(chunks)))

    total_words = sum(result['total_words'] for result in results)
    total_sentences = sum(result['total_sentences'] for result in results)
    unique_words = set().union(*[result['unique_words'] for result in results])
    total_words_without_stopwords = sum(result['total_words_without_stopwords'] for result in results)

    combined_stats = {
        "total_words": total_words,
        "unique_words_count": len(unique_words),
        "total_sentences": total_sentences,
        "total_words_without_stopwords": total_words_without_stopwords
    }

    return combined_stats

def process_and_lemmatize_chunk(chunk):
    """
    Tokenizes and lemmatizes the text in a chunk of the dataframe.

    Args:
        chunk (pandas.DataFrame): A chunk of the dataframe to process.

    Returns:
        pandas.DataFrame: A dataframe containing the lemmatized tokens and their corresponding target values.
    """
    # Tokenize and lemmatize within a single function
    def tokenize_and_lemmatize(text):
        doc = spacy_model(text)
        lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return lemmatized_tokens
    
    chunk['lemmatized_tokens'] = chunk['cleaned_post'].apply(tokenize_and_lemmatize)
    return chunk[['lemmatized_tokens', 'target']]

def tokenize_and_parallelize(train_data, output_parquet, num_chunks=10):
    """
    Tokenizes and lemmatizes the text in the training data and saves the result to a Parquet file.

    Args:
        train_data (pandas.DataFrame): The training data containing the text to tokenize and lemmatize.
        output_parquet (str): The file path where the resulting Parquet file will be saved.
        num_chunks (int, optional): The number of chunks to split the data into for parallel processing. Defaults to 10.
    """
    # Split the train data into chunks for parallel processing
    chunks = np.array_split(train_data, num_chunks)

    # Parallelize the processing (tokenizing and lemmatizing) of each chunk using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        result_chunks = list(executor.map(process_and_lemmatize_chunk, chunks))
    
    # Concatenate the results and save to Parquet
    tokenized_and_lemmatized_train_data = pd.concat(result_chunks, ignore_index=True)
    save_to_parquet_file(tokenized_and_lemmatized_train_data, output_parquet)

def save_to_parquet_file(data, filename):
    """
    Saves the given dataframe to a Parquet file.

    Args:
        data (pandas.DataFrame): The dataframe to save.
        filename (str): The name of the file where the data will be saved.
    """
    table = pa.Table.from_pandas(data)
    pq.write_table(table, filename)