"""
predict_utils.py : Utility functions for loading models, making predictions, and visualizing performance 
metrics for mental health classification using XGBoost and BERT models.

This module provides functionality to:
- Load ONNX models, tokenizers, and encoders
- Predict classes and probabilities from text inputs
- Clean and process model performance reports
- Display comparison charts and ROC curves using Streamlit and Plotly

Dependencies:
    - numpy
    - pandas
    - joblib
    - onnxruntime
    - transformers
    - plotly
    - streamlit
    - PIL (for image display)
"""

import pandas as pd
import streamlit as st
import numpy as np
import joblib
import onnxruntime as ort
from mentalhealth.bert_tiny_modified import clean_text  # Import clean_text function
from transformers import AutoTokenizer
import plotly.express as px

import plotly.graph_objects as go
from PIL import Image

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import torch

torch.set_num_threads(1)


def load_xgb_components(model_path="model/xgb_mental_health.onnx", vec_path="model/tfidf_vectorizer.pkl", enc_path="model/xgb_label_encoder.pkl"):
    """
    Loads the XGBoost model, TF-IDF vectorizer, and label encoder from disk.

    Args:
        model_path (str): Path to the ONNX XGBoost model file.
        vec_path (str): Path to the pickled TF-IDF vectorizer.
        enc_path (str): Path to the pickled label encoder.

    Returns:
        tuple: A tuple containing the ONNX inference session, TF-IDF vectorizer, and label encoder.
    """
    session = ort.InferenceSession(model_path)
    vectorizer = joblib.load(vec_path)
    encoder = joblib.load(enc_path)
    return session, vectorizer, encoder

def predict_with_xgb(texts, session, vectorizer, encoder):
    """
    Predicts mental health classes using the XGBoost ONNX model.

    Args:
        texts (str or List[str]): Input text(s) for classification.
        session (onnxruntime.InferenceSession): Loaded ONNX model session.
        vectorizer (TfidfVectorizer): TF-IDF vectorizer used to transform input texts.
        encoder (LabelEncoder): Label encoder used to decode predicted class indices.

    Returns:
        tuple: Predicted class labels and their corresponding probabilities.
    """
    texts = [texts] if isinstance(texts, str) else texts
    vectors = vectorizer.transform(texts)
    
    input_name = session.get_inputs()[0].name
    inputs = {input_name: vectors.astype(np.float32).toarray()}
    outputs = session.run(None, inputs)[0]

    outputs = np.atleast_2d(outputs)

    if outputs.shape[1] == 1:
        # Output is class index, not probabilities
        preds = outputs[:, 0].astype(int)
        probs = np.zeros((len(preds), len(encoder.classes_)))
        for i, p in enumerate(preds):
            probs[i, p] = 1.0
    else:
        # Output is probabilities
        preds = np.argmax(outputs, axis=1)
        probs = outputs

    return encoder.inverse_transform(preds), probs[0] if len(texts) == 1 else probs


def load_bert_components(
    model_path="model/bert_tiny_model_optimized.onnx",
    tokenizer_path="model/bert_tokenizer",
    encoder_path="model/bert_label_encoder.pkl"
):
    """
    Loads the BERT model, tokenizer, and label encoder from disk.

    Args:
        model_path (str): Path to the ONNX BERT model file.
        tokenizer_path (str): Directory path to the saved tokenizer.
        encoder_path (str): Path to the pickled label encoder.

    Returns:
        tuple: A tuple containing the ONNX inference session, tokenizer, and label encoder.
    """
    session = ort.InferenceSession(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    label_encoder = joblib.load(encoder_path)
    return session, tokenizer, label_encoder

# Prediction function for BERT
def predict_with_bert(texts, session, tokenizer, label_encoder, max_len=128):
    """
    Predicts mental health classes using the BERT ONNX model.

    Args:
        texts (str or List[str]): Input text(s) for classification.
        session (onnxruntime.InferenceSession): Loaded ONNX BERT model session.
        tokenizer (AutoTokenizer): Tokenizer used for input encoding.
        label_encoder (LabelEncoder): Label encoder used to decode predicted class indices.
        max_len (int): Maximum token length for padding/truncation.

    Returns:
        tuple: Predicted class labels and their corresponding probabilities.
    """
    if isinstance(texts, str):
        texts = [texts]

    # Clean and tokenize the texts using the reused clean_text function
    cleaned_texts = [clean_text(text) for text in texts]
    encodings = tokenizer(cleaned_texts, return_tensors="np", padding="max_length", truncation=True, max_length=max_len)
    input_ids = encodings["input_ids"].astype(np.int64)
    attention_mask = encodings["attention_mask"].astype(np.int64)

    # Run prediction on the BERT ONNX model
    logits = session.run(["logits"], {"input_ids": input_ids, "attention_mask": attention_mask})[0]

    # Apply softmax to logits to get probabilities
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)

    decoded_preds = label_encoder.inverse_transform(preds)
    
    return decoded_preds, probs

def clean_and_convert(df, column_names):
    """
    Cleans and converts specified DataFrame columns to numeric types.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_names (List[str]): Names of columns to clean and convert.

    Returns:
        pd.DataFrame: Cleaned and converted DataFrame.
    """
    for col in column_names:
        # Remove extra spaces, unexpected characters, or convert malformed values
        df[col] = df[col].str.replace(r'\D', '', regex=True)  # Remove non-digit characters, if applicable
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, NaN if invalid
    return df


def load_reports():
    """
    Loads model performance reports for XGBoost and BERT from an Excel file.

    Returns:
        tuple: Two pandas DataFrames containing performance metrics for XGBoost and BERT.
    """
    file_path = os.path.join("model","model_reports.xlsx")

    xgb_report = pd.read_excel(file_path, sheet_name="XGBoost")
    bert_report = pd.read_excel(file_path, sheet_name="BERT")

    xgb_report.columns = ['Class', 'XGB Precision', 'XGB Recall', 'XGB F1-Score', 'XGB Support']
    bert_report.columns = ['Class', 'BERT Precision', 'BERT Recall', 'BERT F1-Score', 'BERT Support']
    
    xgb_report = clean_and_convert(xgb_report, xgb_report.columns[1:]) 
    bert_report = clean_and_convert(bert_report, bert_report.columns[1:])
    # Ensure that numeric columns are properly converted to numeric types
    xgb_report['XGB Precision'] = pd.to_numeric(xgb_report['XGB Precision'], errors='coerce')
    xgb_report['XGB Recall'] = pd.to_numeric(xgb_report['XGB Recall'], errors='coerce')
    xgb_report['XGB F1-Score'] = pd.to_numeric(xgb_report['XGB F1-Score'], errors='coerce')
    xgb_report['XGB Support'] = pd.to_numeric(xgb_report['XGB Support'], errors='coerce')

    bert_report['BERT Precision'] = pd.to_numeric(bert_report['BERT Precision'], errors='coerce')
    bert_report['BERT Recall'] = pd.to_numeric(bert_report['BERT Recall'], errors='coerce')
    bert_report['BERT F1-Score'] = pd.to_numeric(bert_report['BERT F1-Score'], errors='coerce')
    bert_report['BERT Support'] = pd.to_numeric(bert_report['BERT Support'], errors='coerce')

    return xgb_report, bert_report

def display_comparison(xgb_report, bert_report):
    """
    Displays a side-by-side comparison of XGBoost and BERT model performance using Streamlit.

    Args:
        xgb_report (pd.DataFrame): Performance metrics for XGBoost.
        bert_report (pd.DataFrame): Performance metrics for BERT.

    Returns:
        None
    """
    st.write("### Model Performance Comparison (XGBoost vs BERT)")

    # Merge the reports on the "Class" column to show a side-by-side comparison
    comparison_df = pd.merge(xgb_report, bert_report, on="Class")
    comparison_df = comparison_df.drop(columns=["XGB Support", "BERT Support"], errors="ignore")

    st.write(comparison_df)

    display_speedometers()

    # Melt the dataframe to long format for plotting
    f1_df = pd.melt(comparison_df, id_vars=["Class"], 
                     value_vars=["XGB F1-Score", "BERT F1-Score"],
                     var_name="Metric", value_name="Value")

    f1_df['Class'] = f1_df['Class'].str.strip()
    
    # Create a bar plot with only F1-Scores
    fig = px.bar(f1_df, x="Class", y="Value", color="Metric", 
            barmode="group", 
             title="Model Performance: XGBoost vs BERT (F1-Score Only)")

    st.plotly_chart(fig)

def display_auc_roc():
    """
    Displays ROC-AUC curve images for BERT and XGBoost models using Streamlit.

    Returns:
        None
    """
    # Paths to the AUC PNGs
    bert_path = os.path.join("roc_curve", "ROC_AUC_BERT.png")
    xgb_path = os.path.join("roc_curve", "ROC_AUC_XGBoost.png")

    auc_images = [bert_path, xgb_path]

    col1, col2 = st.columns(2)
    
    # Display the first image with a heading in the first column
    with col1:
        st.header("BERT ROC Curve")
        img = Image.open(auc_images[0])
        st.image(img, use_container_width=True)
    
    with col2:
        st.header("XGBoost ROC Curve")
        img = Image.open(auc_images[1])
        st.image(img, use_container_width=True)


def create_gauge(title, value, max_value, color):
    """
    Creates a Plotly gauge chart to visualize a metric.

    Args:
        title (str): Title of the gauge.
        value (float): Current value to be displayed.
        max_value (float): Maximum value for the gauge axis.
        color (str): Color of the gauge bar.

    Returns:
        plotly.graph_objects.Figure: Configured gauge chart.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [0, max_value]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, max_value * 0.5], 'color': '#f2f2f2'},
                {'range': [max_value * 0.5, max_value * 0.75], 'color': '#d9d9d9'},
                {'range': [max_value * 0.75, max_value], 'color': '#b3b3b3'}
            ],
        }
    ))
    fig.update_layout(height=250)
    return fig

def display_speedometers():
    """
    Displays gauge charts for accuracy and average F1-score of XGBoost and BERT models using Streamlit.

    Returns:
        None
    """
    st.header("Model Performance Speedometers")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_gauge("XGBoost Accuracy (%)", 79.89, 100, "orange"), use_container_width=True)
    with col2:
        st.plotly_chart(create_gauge("BERT Accuracy (%)", 95.89, 100, "green"), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(create_gauge("XGBoost Avg F1-Score", 63.59, 100, "orange"), use_container_width=True)
    with col4:
        st.plotly_chart(create_gauge("BERT Avg F1-Score", 95.35, 100, "green"), use_container_width=True)
