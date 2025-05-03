import pytest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from mentalhealth.data_visualization import (
    plot_post_counts_by_subreddit,
    plot_post_lengths_by_target,
    generate_wordcloud,
    plot_roc_curve_multiclass
)

def test_plot_post_counts_by_subreddit(sample_data):
    plot_post_counts_by_subreddit(sample_data)
    plt.close()

def test_plot_post_lengths_by_target(sample_data):
    sample_data['post_length'] = sample_data['text'].apply(len)
    plot_post_lengths_by_target(sample_data, x_column='post_length', y_column='subreddit')
    plt.close()

def test_generate_wordcloud(sample_data):
    generate_wordcloud(sample_data, 'text')
    plt.close()

def test_plot_roc_curve_multiclass():
    y_test = np.array([0, 1, 2, 1, 0, 2])
    y_pred = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7],
                       [0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])

    target_names = ['Class 0', 'Class 1', 'Class 2']
    plot_roc_curve_multiclass(y_test, y_pred, classifier_name="Test Classifier", target_names=target_names)
    plt.close()

def test_empty_dataframe():
    empty_data = pd.DataFrame()

    # Test plot post counts by subreddit with empty data
    plot_post_counts_by_subreddit(empty_data)
    plt.close()

    # Test plot post lengths by target with empty data
    plot_post_lengths_by_target(empty_data, x_column='post_length', y_column='target')
    plt.close()

    # Test generate word cloud with empty data
    generate_wordcloud(empty_data, 'text')
    plt.close()

def test_invalid_column():
    invalid_data = pd.DataFrame({
        'subreddit': ['python', 'learnpython'],
        'post_length': [100, 200]
    })

    # Test plot post counts by subreddit with invalid column (should raise KeyError)
    with pytest.raises(KeyError):
        plot_post_counts_by_subreddit(invalid_data, column='non_existing_column')
    
    # Test plot post lengths by target with invalid columns (should raise ValueError)
    with pytest.raises(ValueError):
        plot_post_lengths_by_target(invalid_data, x_column='non_existing_column', y_column='target')

def test_csv_data_load_and_plot(sample_csv_data):
    # Load CSV data and test the functions
    folder = sample_csv_data
    file1 = os.path.join(folder, "file1.csv")
    file2 = os.path.join(folder, "file2.csv")
    
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)
    
    # Test the post counts plot
    plot_post_counts_by_subreddit(data1)
    plot_post_counts_by_subreddit(data2)
    plt.close()

    # Test the post lengths plot
    data1['post_length'] = data1['text'].apply(len)
    data2['post_length'] = data2['text'].apply(len)
    plot_post_lengths_by_target(data1, x_column='post_length', y_column='subreddit')
    plot_post_lengths_by_target(data2, x_column='post_length', y_column='subreddit')
    plt.close()

    # Test word cloud generation
    generate_wordcloud(data1, 'text')
    generate_wordcloud(data2, 'text')
    plt.close()

def test_missing_columns():
    # Creating a DataFrame without the 'post_length' and 'target' columns
    data_without_columns = pd.DataFrame({
        'subreddit': ['anxiety', 'fitness', 'bipolarreddit', 'jokes', 'healthanxiety'],
        'text': ['Feeling anxious', 'Gym time!', 'Struggling with bipolar', 'Why did the chicken...', 'Health worries.']
    })
    
    # Test the plot function with missing columns (should raise an error)
    with pytest.raises(ValueError):
        plot_post_lengths_by_target(data_without_columns, x_column='post_length', y_column='target')
