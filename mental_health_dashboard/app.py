"""
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import torch

from mentalhealth.predict_utils import (
    load_xgb_components, predict_with_xgb,
    load_bert_components, predict_with_bert
)

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

    st.markdown("---")
    st.markdown("Made with ❤️ using BERT and XGBoost for mental health awareness.")


# Run the Streamlit app directly
if __name__ == "__main__":
    run_app()
