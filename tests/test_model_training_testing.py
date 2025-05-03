import pytest
import pandas as pd
import numpy as np
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mentalhealth.model_training_testing import (
    extract_tfidf_features, compute_performance_metrics,
    train_logistic_model
)
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import lightgbm as lgb

@pytest.fixture
def sample_data():
    # Create some sample data for testing
    X, y = make_classification(n_samples=100, n_features=10, n_classes=3, n_informative=3, n_clusters_per_class=2, random_state=42)
    return X, y

def test_extract_tfidf_features(sample_data):
    X,_ = sample_data
    X_train_text = ["sample document {}".format(i) for i in range(len(X))]
    
    tfidf_matrix, vectorizer = extract_tfidf_features(X_train_text)
    
    assert tfidf_matrix.shape[0] == len(X_train_text), "TF-IDF matrix has wrong number of rows"
    assert isinstance(vectorizer, TfidfVectorizer), "Expected TfidfVectorizer object"
    assert tfidf_matrix.shape[1] > 0, "TF-IDF matrix has no features"


def test_compute_performance_metrics():
    # Check if performance metrics are computed correctly
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    
    metrics = compute_performance_metrics(y_true, y_pred)
    
    # Check if accuracy, precision, recall, and F1 score are calculated
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert isinstance(metrics['classification_report'], str)


def test_train_logistic_model(sample_data):
    X, y = sample_data
    model = train_logistic_model(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.5, f"Model accuracy is too low: {accuracy}"

def test_train_xgb_model(sample_data):
    X, y = sample_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(eval_metric='mlogloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    assert accuracy > 0.5, f"Model accuracy is too low: {accuracy}"


def test_train_lgbm_model(sample_data):
    X, y = sample_data
    X = pd.DataFrame(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    assert accuracy > 0.5, f"Model accuracy is too low: {accuracy}"

def test_save_model(sample_data, tmpdir):
    X, y = sample_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    model_path = tmpdir.join("model.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    assert os.path.exists(model_path), "Model file not found"

    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)
    
    accuracy = loaded_model.score(X_test, y_test)
    assert accuracy > 0.5, f"Model accuracy is too low: {accuracy}"


def test_train_and_evaluate_model(sample_data):
    X, y = sample_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    assert accuracy > 0.5, f"Model accuracy is too low: {accuracy}"

@pytest.mark.parametrize("model_type", ['logistic', 'xgb', 'lgbm'])
def test_model_selection(sample_data, model_type):
    X, y = sample_data
    
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'logistic':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)  # Use max_iter to ensure convergence
    elif model_type == 'xgb':
        import xgboost as xgb
        model = xgb.XGBClassifier()
    elif model_type == 'lgbm':
        model = lgb.LGBMClassifier()

    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    assert accuracy > 0.5, f"Model accuracy is too low: {accuracy}"

