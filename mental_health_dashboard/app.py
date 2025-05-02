"""
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import torch
from PIL import Image

st.set_page_config(layout="wide", page_title=" Mental Health Text Classifier")

st.markdown(
    """
    <style>
        .block-container {
            max-width: 95% !important;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .css-1d391kg {
            width: 300px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

from mentalhealth.predict_utils import (
    load_xgb_components, predict_with_xgb,
    load_bert_components, predict_with_bert,
    load_reports, display_comparison,
    display_auc_roc
)

def display_wordcloud(class_name):
    # Path to the pre-generated word clouds (This is a placeholder path, update as per your actual file locations)
    wordcloud_path = os.path.join("wordclouds", f"WordCloud_{class_name}.png")
    if os.path.exists(wordcloud_path):
        img = Image.open(wordcloud_path)
        st.image(img, caption=f"Word Cloud for {class_name}", use_container_width=True)
    else:
        st.error(f"No word cloud found for class: {class_name}")

def run_app():

    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

    st.title("🧠 Mental Health Text Classifier")

    # --- Paths to models ---
    bert_model_path = os.path.join("onnx", "bert_tiny_model_optimized.onnx")
    bert_tokenizer_path = os.path.join("model", "bert_tiny_tokenizer")
    bert_encoder_path = os.path.join("model", "bert_tiny_label_encoder.pkl")

    xgb_model_path = os.path.join("model", "xgb_mental_health.onnx")
    xgb_vectorizer_path = os.path.join("model", "tfidf_vectorizer.pkl")
    xgb_encoder_path = os.path.join("model", "xgb_label_encoder.pkl")

    # --- Load models ---
    bert_session, bert_tokenizer, bert_encoder = load_bert_components(
        model_path=bert_model_path,
        tokenizer_path=bert_tokenizer_path,
        encoder_path=bert_encoder_path
    )
    xgb_session, xgb_vectorizer, xgb_encoder = load_xgb_components(
        xgb_model_path, xgb_vectorizer_path, xgb_encoder_path
    )

    # --- Sidebar ---
    with st.sidebar:
        st.header("Model Settings")
        model_choice = st.radio("Select model:", ["BERT (Transformer)", "XGBoost (Traditional ML)"])
        view_choice = st.sidebar.radio("Select view:", ["Prediction", "Model Comparison", "Class Distribution", "Word Cloud"])

    # --- User Input ---
    st.subheader("Enter a mental health-related message:")
    user_input = st.text_area("Text Input", placeholder="Type or paste text here...", height=200)

    # --- Prediction Trigger ---
    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            if model_choice == "BERT (Transformer)":
                pred, probs = predict_with_bert(user_input, bert_session, bert_tokenizer, bert_encoder)
                labels = bert_encoder.classes_
            else:
                pred, probs = predict_with_xgb(user_input, xgb_session, xgb_vectorizer, xgb_encoder)
                labels = xgb_encoder.classes_

            st.success(f"**Predicted Mental Health Condition:** `{pred[0]}`")

            # Probability bar chart
            st.subheader("Prediction Probabilities")
            probs = probs.tolist() if isinstance(probs, torch.Tensor) else probs

            if isinstance(labels, np.ndarray):
                labels = labels.tolist()
            
            # Ensure probs is flat
            if isinstance(probs, list) and isinstance(probs[0], list):
                probs = probs[0]
            elif isinstance(probs, torch.Tensor):
                probs = probs.squeeze().tolist()

            # Ensure labels is a list of strings
            if not isinstance(labels, list):
                labels = labels.tolist()

            # Sanity check: flatten probs if nested
            if isinstance(probs, (list, np.ndarray)) and np.ndim(probs) > 1:
                probs = probs[0]

            # Final shape check
            assert len(labels) == len(probs), f"Length mismatch: labels={len(labels)}, probs={len(probs)}"

            prob_df = pd.DataFrame({
                "Class": labels,
                "Probability": probs if isinstance(probs, list) else probs.tolist()
            }).sort_values("Probability", ascending=False)

            st.bar_chart(prob_df.set_index("Class"))

    if view_choice == "Model Comparison":

        st.header("ROC Comparison")
        display_auc_roc()
    
    elif view_choice == "Class Distribution":

        xgb_report, bert_report = load_reports()
        display_comparison(xgb_report, bert_report)
        
    elif view_choice == "Word Cloud":
        
                # Let users choose the class for the word cloud
        classes = ['COVID19_support', 'EDAnonymous', 'Addiction', 'ADHD',
       'alcoholism', 'Anxiety', 'Autism', 'Bipolar', 'bpd',
       'depression', 'healthanxiety', 'Lonely', 'mentalhealth',
       'PTSD', 'schizophrenia', 'socialanxiety',
       'SuicideWatch']
        
        selected_class = st.selectbox("Select a class to view the word cloud", classes)
        
        display_wordcloud(selected_class)

    st.markdown("---")
    st.markdown("Made using BERT and XGBoost for mental health awareness.")


if __name__ == "__main__":
    run_app()
