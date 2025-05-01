"""
"""

import numpy as np
import joblib
import onnxruntime as ort
from mentalhealth.bert_tiny_modified import clean_text  # Import clean_text function
from transformers import AutoTokenizer

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import torch

torch.set_num_threads(1)


def load_xgb_components(model_path="model/xgb_mental_health.onnx", vec_path="model/tfidf_vectorizer.pkl", enc_path="model/xgb_label_encoder.pkl"):
    session = ort.InferenceSession(model_path)
    vectorizer = joblib.load(vec_path)
    encoder = joblib.load(enc_path)
    return session, vectorizer, encoder

def predict_with_xgb(texts, session, vectorizer, encoder):
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
    session = ort.InferenceSession(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    label_encoder = joblib.load(encoder_path)
    return session, tokenizer, label_encoder

# Prediction function for BERT
def predict_with_bert(texts, session, tokenizer, label_encoder, max_len=128):
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
