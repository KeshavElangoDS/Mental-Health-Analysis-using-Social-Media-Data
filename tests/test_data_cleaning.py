# tests/test_data_cleaning.py
import pytest
import pandas as pd
from mentalhealth.data_cleaning import (
    load_and_combine_dataset,
    label_mental_health_subreddits,
    clean_text
)

def test_load_and_combine_dataset_valid(sample_csv_data):
    df = load_and_combine_dataset(str(sample_csv_data))
    assert not df.empty
    assert df.shape[1] == 4
    assert 'subreddit' in df.columns

def test_load_and_combine_dataset_empty(tmpdir):
    empty_dir = tmpdir.mkdir("empty")
    df = load_and_combine_dataset(str(empty_dir))
    assert df.empty

def test_label_mental_health_subreddits_valid(sample_data):
    df, subreddits_df = label_mental_health_subreddits(sample_data)
    assert 'target' in df.columns
    assert set(df['target']) <= {'anxiety', 'fitness', 'bipolarreddit', 'healthanxiety', 'non_mental_health'}

def test_label_mental_health_subreddits_missing_column():
    bad_df = pd.DataFrame({'topic': ['anxiety', 'fitness']})
    with pytest.raises(KeyError):
        label_mental_health_subreddits(bad_df)

def test_label_mental_health_subreddits_empty():
    df = pd.DataFrame(columns=['subreddit'])
    result, _ = label_mental_health_subreddits(df)
    assert result.empty

def test_clean_text_ascii():
    assert clean_text('Just ASCII!') == 'Just ASCII!'

def test_clean_text_with_unicode():
    assert clean_text('Café 😊') == 'Caf '

def test_clean_text_html_entity():
    assert clean_text('Fish &amp; Chips') == 'Fish & Chips'

def test_clean_text_empty():
    assert clean_text('') == ''
