"""
model_training_testing.py : This module contains functions for training, evaluating, and exporting machine learning models 
(Logistic Regression, Naive Bayes, XGBoost, and LightGBM) for text classification using TF-IDF features.

It supports model training, performance evaluation, ONNX export, and preprocessing of tokenized text data.

Functions:
    - extract_tfidf_features: Converts raw text into TF-IDF feature vectors.
    - compute_performance_metrics: Calculates classification metrics including accuracy, precision, recall, and F1 score.
    - train_logistic_model: Trains a Logistic Regression classifier.
    - train_nb_model: Trains a Multinomial Naive Bayes classifier.
    - train_xgb_model: Trains an XGBoost classifier with label encoding.
    - train_lgbm_model: Trains a LightGBM classifier with class imbalance handling.
    - save_model: Saves trained models in ONNX format or native LightGBM format.
    - train_and_evaluate_model: Orchestrates model training, prediction, and evaluation.
    - display_performance_metrics: Prints and returns key evaluation metrics for model performance.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import re

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxmltools

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB
import lightgbm as lgb

import os, joblib

# Function to apply TF-IDF for feature extraction
def extract_tfidf_features(text_data, max_features=5000):
    """
    Extract TF-IDF features from the provided text data.

    Args:
        text_data (array-like): Input text data, either a list or an ndarray of documents.
        max_features (int): The maximum number of features (default is 5000).

    Returns:
        tuple: 
            - tfidf_matrix (sparse matrix): TF-IDF feature matrix.
            - tfidf_vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer used for transformation.
    """
    if isinstance(text_data, np.ndarray):
        text_data = text_data.flatten()  # Flatten if multi-dimensional

    text_data = [str(doc) for doc in text_data]

    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
    return tfidf_matrix, tfidf_vectorizer

# Function to compute performance metrics
def compute_performance_metrics(y_test, y_pred):
    """
    Compute performance metrics for classification tasks.

    Args:
        y_test (array-like): True labels of the test set.
        y_pred (array-like): Predicted labels for the test set.

    Returns:
        dict: A dictionary containing the accuracy, precision, recall, F1 score, and classification report.
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
    
    report = classification_report(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report
    }


def train_logistic_model(X_train_tfidf, y_train):
    """
    Train a logistic regression model on the given TF-IDF training data.

    Args:
        X_train_tfidf (sparse matrix): The TF-IDF feature matrix for training.
        y_train (array-like): The target labels for training.

    Returns:
        model (LogisticRegression): The trained Logistic Regression model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    return model


def train_nb_model(X_train_tfidf, y_train):
    """
    Train a Naive Bayes model on the given TF-IDF training data.

    Args:
        X_train_tfidf (sparse matrix): The TF-IDF feature matrix for training.
        y_train (array-like): The target labels for training.

    Returns:
        model (MultinomialNB): The trained Naive Bayes model.
    """
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    return model


def train_xgb_model(X_train_tfidf, y_train):
    """
    Train an XGBoost model on the given TF-IDF training data.

    Args:
        X_train_tfidf (sparse matrix): The TF-IDF feature matrix for training.
        y_train (array-like): The target labels for training.

    Returns:
        tuple: 
            - model (XGBClassifier): The trained XGBoost model.
            - label_encoder (LabelEncoder): The label encoder used to encode target labels.
    """
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    model = xgb.XGBClassifier(use_label_encoder=False)
    model.fit(X_train_tfidf, y_train_encoded)
    
    return model, label_encoder


def train_lgbm_model(X_train_tfidf, y_train, n_classes=3):
    """
    Train a LightGBM model on the given TF-IDF training data.

    Args:
        X_train_tfidf (sparse matrix): The TF-IDF feature matrix for training.
        y_train (array-like): The target labels for training.
        n_classes (int): The number of classes in the target labels (default is 3).

    Returns:
        model (LGBMClassifier): The trained LightGBM model.
    """
    model = lgb.LGBMClassifier(scale_pos_weight=10, n_estimators=50, max_depth=5) #scale_pos_weight=10 for class imbalance
    # For multi-class, create an init_score array with the shape (n_samples, n_classes)
    init_score = np.full((len(y_train), n_classes), 0.5, dtype=float)
    model.fit(X_train_tfidf, y_train, init_score=init_score)
    return model

def save_model(model, save_as, model_filename, X_train):
    """
    Save the trained model to disk in the specified format.

    Args:
        model (object): The trained model to save.
        save_as (str): The format to save the model in (e.g., 'onnx').
        model_filename (str): The filename to save the model to.
        X_train (array-like): The training data used for fitting the model (needed for ONNX export).
    """
    if save_as == 'onnx':
        if isinstance(model, xgb.XGBClassifier):  # For XGBoost models
            onnx_model = onnxmltools.convert.convert_xgboost(model, initial_types=[('input', FloatTensorType([None, X_train.shape[1]]))])
            with open(model_filename, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            print(f"Model saved as ONNX at {model_filename}")

        elif isinstance(model, lgb.LGBMClassifier):  # For LightGBM models
            booster = model.booster_
            booster.save_model(model_filename)
        
        else:
            onnx_model = convert_sklearn(model, initial_types=[('input', FloatTensorType([None, X_train.shape[1]]))])
            with open(model_filename, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            print(f"Model saved as ONNX at {model_filename}")

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type='logistic'):
    """
    Train and evaluate a model on the given training and testing data.

    Args:
        X_train (array-like): The training feature data.
        y_train (array-like): The training labels.
        X_test (array-like): The testing feature data.
        y_test (array-like): The testing labels.
        model_type (str): The type of model to train ('logistic', 'nb', 'xgb', 'lgbm').

    Returns:
        tuple: 
            - y_test (array-like): True labels of the test set.
            - y_pred (array-like): Predicted labels for the test set.
            - ypred_proba (array-like): Predicted probabilities for the test set.
            - model_name (str): The name of the model.
            - target_names (array-like): The unique class labels in the target variable.
    """
    # Preprocess training and testing data
    X_train = X_train.apply(lambda tokens: ' '.join([token for token in tokens if not re.search(r'\d', token) and '/w' not in token]))
    X_test = X_test.apply(lambda tokens: ' '.join([token for token in tokens if not re.search(r'\d', token) and '/w' not in token]))

    X_train_tfidf, tfidf_vectorizer = extract_tfidf_features(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    os.makedirs("model", exist_ok=True)
    joblib.dump(tfidf_vectorizer, "model/tfidf_vectorizer.pkl")

    target_names = np.unique(y_train)

    # Model selection and training
    if model_type == 'logistic':
        model = train_logistic_model(X_train_tfidf, y_train)
        model_name = 'Logistic Regression'
        model_filename = 'model/logistic_mental_health.onnx'
    elif model_type == 'nb':
        model = train_nb_model(X_train_tfidf, y_train)
        model_name = 'Naive Bayes'
        model_filename = 'model/nb_mental_health.onnx'
    elif model_type == 'xgb':
        model, label_encoder = train_xgb_model(X_train_tfidf, y_train)
        model_name = 'XGBoost'
        model_filename = 'model/xgb_mental_health.onnx'

        # Save label encoder
        joblib.dump(label_encoder, "model/xgb_label_encoder.pkl")
    elif model_type == 'lgbm':
        model = train_lgbm_model(X_train_tfidf, y_train, n_classes = len(y_train.unique()))
        model_name = 'LightGBM'
        model_filename = 'model/lgbm_mental_health.onnx'
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    save_model(model, 'onnx', model_filename, X_train_tfidf)

    y_pred_encoded = model.predict(X_test_tfidf)
    y_pred = label_encoder.inverse_transform(y_pred_encoded) if model_type == 'xgb' else y_pred_encoded

    ypred_proba = model.predict_proba(X_test_tfidf)

    return y_test, y_pred, ypred_proba, model_name, target_names


def display_performance_metrics(y_test, y_pred, model_name):
    """
    Display the performance metrics of a trained model.

    Args:
        y_test (array-like): True labels of the test set.
        y_pred (array-like): Predicted labels for the test set.
        model_name (str): The name of the model.

    Returns:
        dict: A dictionary containing the computed performance metrics.
    """
    metrics = compute_performance_metrics(y_test, y_pred)
    print(f"Model: {model_name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("Classification Report:")
    print(metrics['classification_report'])
    print('-' * 50)

    return metrics
