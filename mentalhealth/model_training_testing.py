"""
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
# from concurrent.futures import ProcessPoolExecutor
import re

import pickle
from skl2onnx import convert_sklearn
from onnxmltools.convert import convert_lightgbm
from skl2onnx.common.data_types import FloatTensorType
import onnxmltools

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB
import lightgbm as lgb

# Function to apply TF-IDF for feature extraction
def extract_tfidf_features(text_data, max_features=5000):
    if isinstance(text_data, np.ndarray):
        text_data = text_data.flatten()  # Flatten if multi-dimensional

    text_data = [str(doc) for doc in text_data]  # Ensure all elements are strings

    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
    return tfidf_matrix, tfidf_vectorizer

# Function to compute performance metrics
def compute_performance_metrics(y_test, y_pred):
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

# Function to train a logistic regression model
def train_logistic_model(X_train_tfidf, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    return model

# Function to train a Naive Bayes model
def train_nb_model(X_train_tfidf, y_train):
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    return model

# Function to train an XGBoost model

def train_xgb_model(X_train_tfidf, y_train):    
    # Encode the string labels into numeric labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    model = xgb.XGBClassifier(use_label_encoder=False)
    model.fit(X_train_tfidf, y_train_encoded)
    
    return model, label_encoder

# Function to train a LightGBM model
def train_lgbm_model(X_train_tfidf, y_train, n_classes=3):
    model = lgb.LGBMClassifier(scale_pos_weight=10, n_estimators=50, max_depth=5) #scale_pos_weight=10 for class imbalance
    # For multi-class, create an init_score array with the shape (n_samples, n_classes)
    init_score = np.full((len(y_train), n_classes), 0.5, dtype=float)
    model.fit(X_train_tfidf, y_train, init_score=init_score)
    return model

def save_model(model, save_as, model_filename, X_train):
    if save_as == 'onnx':
        if isinstance(model, xgb.XGBClassifier):  # For XGBoost models
            # Convert XGBoost model to ONNX format
            onnx_model = onnxmltools.convert.convert_xgboost(model, initial_types=[('input', FloatTensorType([None, X_train.shape[1]]))])
            with open(model_filename, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            print(f"Model saved as ONNX at {model_filename}")

        elif isinstance(model, lgb.LGBMClassifier):  # For LightGBM models
            # Convert LightGBM model to ONNX format
            # Save model to a file
            booster = model.booster_
            
            # Save the Booster model to a file
            booster.save_model(model_filename)

            # # Load the model back
            # loaded_model = lgb.Booster(model_file='lightgbm_model.txt')

            # # Predict using the loaded model
            # y_pred = loaded_model.predict(X_test)
        
        else:
            # Handle other model types, e.g., for scikit-learn models
            onnx_model = convert_sklearn(model, initial_types=[('input', FloatTensorType([None, X_train.shape[1]]))])
            with open(model_filename, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            print(f"Model saved as ONNX at {model_filename}")

# # Function to save the model in the specified format
# def save_model(model, save_as, model_filename, X_train):
#     if save_as == 'pickle':
#         with open(model_filename, 'wb') as f:
#             pickle.dump(model, f)
#         print(f"Model saved as Pickle at {model_filename}")
#     elif save_as == 'onnx':
#         onnx_model = convert_sklearn(model, initial_types=[('input', FloatTensorType([None, X_train.shape[1]]))])
#         with open(model_filename, 'wb') as f:
#             f.write(onnx_model.SerializeToString())
#         print(f"Model saved as ONNX at {model_filename}")
#     else:
#         raise ValueError(f"Save format {save_as} is not supported.")

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type='logistic'):
    # Preprocess training and testing data (removing digits and unwanted tokens)
    X_train = X_train.apply(lambda tokens: ' '.join([token for token in tokens if not re.search(r'\d', token) and '/w' not in token]))
    X_test = X_test.apply(lambda tokens: ' '.join([token for token in tokens if not re.search(r'\d', token) and '/w' not in token]))

    # Apply TF-IDF feature extraction
    X_train_tfidf, tfidf_vectorizer = extract_tfidf_features(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Select model based on the model_type argument
    if model_type == 'logistic':
        model = train_logistic_model(X_train_tfidf, y_train)
        model_name = 'Logistic Regression'
        model_filename = 'logistic_mental_health.onnx'
    elif model_type == 'nb':
        model = train_nb_model(X_train_tfidf, y_train)
        model_name = 'Naive Bayes'
        model_filename = 'nb_mental_health.onnx'
    elif model_type == 'xgb':
        model, label_encoder = train_xgb_model(X_train_tfidf, y_train)
        model_name = 'XGBoost'
        model_filename = 'xgb_mental_health.onnx'
    elif model_type == 'lgbm':
        model = train_lgbm_model(X_train_tfidf, y_train, n_classes = len(y_train.unique()))
        model_name = 'LightGBM'
        model_filename = 'lgbm_mental_health.onnx'
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    save_model(model, 'onnx', model_filename, X_train_tfidf)

    # Predict and decode the predictions back to original labels
    y_pred_encoded = model.predict(X_test_tfidf)
    if model_type == 'xgb':
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
    else:
        y_pred = y_pred_encoded

    return y_test, y_pred, model_name


# Function to display performance metrics
def display_performance_metrics(y_test, y_pred, model_name):
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


