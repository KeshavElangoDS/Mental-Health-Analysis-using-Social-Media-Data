"""
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

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer, BertForSequenceClassification

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
            # Save model to a file
            booster = model.booster_
            
            # Save the Booster model to a file
            booster.save_model(model_filename)
        
        else:
            # Handle other model types, e.g., for scikit-learn models
            onnx_model = convert_sklearn(model, initial_types=[('input', FloatTensorType([None, X_train.shape[1]]))])
            with open(model_filename, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            print(f"Model saved as ONNX at {model_filename}")

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type='logistic'):
    # Preprocess training and testing data (removing digits and unwanted tokens)
    X_train = X_train.apply(lambda tokens: ' '.join([token for token in tokens if not re.search(r'\d', token) and '/w' not in token]))
    X_test = X_test.apply(lambda tokens: ' '.join([token for token in tokens if not re.search(r'\d', token) and '/w' not in token]))

    # Apply TF-IDF feature extraction
    X_train_tfidf, tfidf_vectorizer = extract_tfidf_features(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    target_names = np.unique(y_train)

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

    # Predict probabilities (added as ypred_proba)
    ypred_proba = model.predict_proba(X_test_tfidf)

    return y_test, y_pred, ypred_proba, model_name, target_names


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



def tokenize_bert_batch(texts, tokenizer, batch_size=32, max_length=128):
    """
    Tokenize the text data into input_ids and attention masks for BERT.
    Ensure uniform padding and truncation.
    """
    input_ids = []
    attention_masks = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenizing with padding and truncation to max_length
        encoded_batch = tokenizer(
            batch_texts,
            padding='max_length',  # Ensure padding to max_length
            truncation=True,  # Truncate sequences longer than max_length
            max_length=max_length,  # Limit the length of the sequences
            return_tensors='pt'
        )
        
        # Debugging: Check the shape of input_ids and attention_masks
        print(f"Batch {i // batch_size}:")
        print(f"  input_ids shape: {encoded_batch['input_ids'].shape}")
        print(f"  attention_mask shape: {encoded_batch['attention_mask'].shape}")
        
        input_ids.append(encoded_batch['input_ids'])
        attention_masks.append(encoded_batch['attention_mask'])
    
    # Concatenate the lists of input_ids and attention_masks
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    return input_ids, attention_masks

def prepare_bert_data(mental_health_df, tokenizer, batch_size=32):
    """
    Prepare the data for training by tokenizing the text and splitting into training and test sets.
    """
    texts, labels = mental_health_df['cleaned_post'].tolist(), mental_health_df['target'].tolist()

    # Convert string labels to numeric labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)  # Convert string labels to integers
    
    input_ids, attention_masks = tokenize_bert_batch(texts, tokenizer, batch_size)
    
    # Convert labels to tensors
    labels_tensor = torch.tensor(labels)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(input_ids, labels_tensor, test_size=0.2, random_state=42)
    
    # Create PyTorch DataLoader
    train_dataset = TensorDataset(X_train, attention_masks[:len(X_train)], y_train)
    test_dataset = TensorDataset(X_test, attention_masks[len(X_train):], y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, tokenizer, labels_tensor

def load_bert_model(num_labels):
    """
    Load the pre-trained BERT model for sequence classification.
    """
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    return model

def train_bert_model(model, train_loader, optimizer, device, epochs=3):
    """
    Train the BERT model for the specified number of epochs.
    """
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            batch_input_ids, batch_attention_masks, batch_labels = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels)
            
            # Compute loss and backpropagate
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1} complete. Loss: {loss.item()}')

def evaluate_bert_model(model, test_loader, device):
    """
    Evaluate the model on the test dataset.
    """
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            batch_input_ids, batch_attention_masks, batch_labels = [b.to(device) for b in batch]
            
            # Forward pass
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks)
            
            # Get predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            predictions.extend(preds)
            true_labels.extend(batch_labels.cpu().numpy())
    
    # Print classification metrics
    print(classification_report(true_labels, predictions))

def train_and_evaluate_bert_model(dataset_filepath):
    """
    """
    mental_health_df = pd.read_csv(dataset_filepath)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_loader, test_loader, tokenizer, labels_tensor = prepare_bert_data(mental_health_df, tokenizer)
    
    num_labels = len(set(labels_tensor))  # Number of unique classes
    model = load_bert_model(num_labels)
    
    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)
    
    # Set up device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    train_bert_model(model, train_loader, optimizer, device, epochs=3)
    evaluate_bert_model(model, test_loader, device)
    
    model.save_pretrained('mental_health_bert_model')
    tokenizer.save_pretrained('mental_health_bert_model')