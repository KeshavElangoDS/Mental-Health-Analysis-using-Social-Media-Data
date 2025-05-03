# tests/conftest.py
import pytest
import pandas as pd

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'subreddit': ['anxiety', 'fitness', 'bipolarreddit', 'jokes', 'healthanxiety'],
        'text': ['Feeling anxious', 'Gym time!', 'Struggling with bipolar', 'Why did the chicken...', 'Health worries.']
    })

@pytest.fixture
def sample_csv_data(tmpdir):
    data1 = {
        'subreddit': ['anxiety', 'bipolarreddit'],
        'text': ['I am anxious', 'Bipolar is tough'],
        'extra_col_1': ['x1', 'x2'],
        'extra_col_2': ['x3', 'x4'],
    }
    data2 = {
        'subreddit': ['fitness', 'healthanxiety'],
        'text': ['Exercise matters', 'Health anxiety is real'],
        'extra_col_1': ['x5', 'x6'],
        'extra_col_2': ['x7', 'x8'],
    }

    folder = tmpdir.mkdir("csvs")
    pd.DataFrame(data1).to_csv(folder.join("file1.csv"), index=False)
    pd.DataFrame(data2).to_csv(folder.join("file2.csv"), index=False)

    return folder
