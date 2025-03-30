"""
"""

import spacy
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import math

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np


spacy_model = spacy.load('en_core_web_sm')

# Natural Language Processing with Python and spaCy: A Practical Introduction by Yuli Vasiliev

def text_statistics(df, text_column):
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
    return text_statistics(chunk, text_column)

def parallel_text_statistics(df, text_column, num_cores=3):
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
    # Tokenize and lemmatize within a single function
    def tokenize_and_lemmatize(text):
        doc = spacy_model(text)
        lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return lemmatized_tokens
    
    chunk['lemmatized_tokens'] = chunk['cleaned_post'].apply(tokenize_and_lemmatize)
    return chunk[['lemmatized_tokens', 'target']]

def tokenize_and_parallelize(train_data, output_parquet, num_chunks=10):
    # Split the train data into chunks for parallel processing
    chunks = np.array_split(train_data, num_chunks)

    # Parallelize the processing (tokenizing and lemmatizing) of each chunk using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        result_chunks = list(executor.map(process_and_lemmatize_chunk, chunks))
    
    # Concatenate the results and save to Parquet
    tokenized_and_lemmatized_train_data = pd.concat(result_chunks, ignore_index=True)
    save_to_parquet_file(tokenized_and_lemmatized_train_data, output_parquet)

def save_to_parquet_file(data, filename):
    table = pa.Table.from_pandas(data)
    pq.write_table(table, filename)