import pandas as pd
import numpy as np
import os
from mentalhealth import data_preprocessing as dp

def test_text_statistics_real_input():
    df = pd.DataFrame({'text': ["This is a test sentence.", "Another test."]})
    stats = dp.text_statistics(df, 'text')
    
    assert isinstance(stats, dict)
    assert stats['total_words'] > 0
    assert stats['total_sentences'] >= 2
    assert isinstance(stats['unique_words'], set)
    assert stats['total_words_without_stopwords'] <= stats['total_words']


def test_process_chunk_real_input():
    df = pd.DataFrame({'text': ["A quick brown fox.", "Jumps over the lazy dog."]})
    stats = dp.process_chunk(df, 'text')
    
    assert isinstance(stats, dict)
    assert stats['total_words'] > 0


def test_parallel_text_statistics_real_input():
    df = pd.DataFrame({'text': ["Sentence one.", "Sentence two.", "Sentence three.", "Sentence four."]})
    stats = dp.parallel_text_statistics(df, 'text', num_cores=2)
    
    assert 'total_words' in stats
    assert 'unique_words_count' in stats
    assert 'total_sentences' in stats
    assert stats['total_sentences'] > 0


def test_process_and_lemmatize_chunk_real_input():
    df = pd.DataFrame({'cleaned_post': ["Cats are running faster than dogs."], 'target': [1]})
    result = dp.process_and_lemmatize_chunk(df)

    assert 'lemmatized_tokens' in result.columns
    assert isinstance(result['lemmatized_tokens'].iloc[0], list)
    assert 'target' in result.columns


def test_tokenize_and_parallelize_real_input(tmp_path):
    df = pd.DataFrame({
        'cleaned_post': ["I love natural language processing.", "SpaCy makes NLP easy."],
        'target': [0, 1]
    })
    output_file = tmp_path / "test_output.parquet"

    dp.tokenize_and_parallelize(df, str(output_file), num_chunks=2)

    assert output_file.exists()

    # Check content of the saved file
    loaded = pd.read_parquet(output_file)
    assert 'lemmatized_tokens' in loaded.columns
    assert 'target' in loaded.columns
    assert isinstance(loaded['lemmatized_tokens'].iloc[0], (list, np.ndarray))


def test_save_to_parquet_file_real_input(tmp_path):
    df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    filepath = tmp_path / "test_file.parquet"
    
    dp.save_to_parquet_file(df, str(filepath))
    assert os.path.exists(filepath)

    loaded = pd.read_parquet(filepath)
    assert df.equals(loaded)
