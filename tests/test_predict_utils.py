import os
import pytest
import onnxruntime as ort
import joblib
import numpy as np
import pandas as pd

from mentalhealth.predict_utils import (
    load_xgb_components, predict_with_xgb,
    load_bert_components, predict_with_bert,
    clean_and_convert, load_reports,
    display_comparison, display_auc_roc,
    create_gauge
)


@pytest.fixture
def model_files():
    model_path = "model/xgb_mental_health.onnx"
    vec_path = "model/tfidf_vectorizer.pkl"
    enc_path = "model/xgb_label_encoder.pkl"
    
    assert os.path.exists(model_path), "Model file missing!"
    assert os.path.exists(vec_path), "Vectorizer file missing!"
    assert os.path.exists(enc_path), "Encoder file missing!"
    
    return model_path, vec_path, enc_path

def test_load_xgb_components(model_files):
    model_path, vec_path, enc_path = model_files
    
    session, vectorizer, encoder = load_xgb_components(model_path, vec_path, enc_path)
    
    assert isinstance(session, ort.InferenceSession), "Failed to load XGBoost session"
    
    assert hasattr(vectorizer, 'transform'), "Failed to load TF-IDF Vectorizer"
    assert hasattr(encoder, 'inverse_transform'), "Failed to load label encoder"

def test_predict_with_xgb(model_files, sample_data):
    model_path, vec_path, enc_path = model_files
    session, vectorizer, encoder = load_xgb_components(model_path, vec_path, enc_path)

    texts = ["Feeling anxious"]
    preds, probs = predict_with_xgb(texts, session, vectorizer, encoder)

    assert len(preds) == 1, "Prediction should return one result"
    assert isinstance(preds[0], str), "Predicted class should be a string"

    assert isinstance(probs, np.ndarray), "Probabilities should be a numpy array"
    
    if probs.ndim == 1:
        assert probs.shape[0] > 0, "Probabilities array should have at least one element"
    elif probs.ndim == 2:
        assert probs.shape[1] > 0, "Probabilities should have at least one column"
    else:
        assert False, "Unexpected number of dimensions for probabilities"


def test_load_reports():
    xgb_report, bert_report = load_reports()
    
    assert isinstance(xgb_report, pd.DataFrame), "XGBoost report should be a DataFrame"
    assert isinstance(bert_report, pd.DataFrame), "BERT report should be a DataFrame"
    assert 'Class' in xgb_report.columns, "XGBoost report missing 'Class' column"
    assert 'Class' in bert_report.columns, "BERT report missing 'Class' column"

def test_display_comparison(model_files, sample_data):
    xgb_report, bert_report = load_reports()
    try:
        display_comparison(xgb_report, bert_report)
    except Exception as e:
        pytest.fail(f"Display Comparison failed with error: {e}")

def test_display_auc_roc():
    try:
        display_auc_roc()
    except Exception as e:
        pytest.fail(f"Display ROC failed with error: {e}")


def test_create_gauge():
    fig = create_gauge("Accuracy", 80, 100, "green")

    assert fig is not None, "Gauge chart should not be None"
    assert 'indicator' in fig['data'][0]['type'], "Figure should be an indicator chart"
    assert 'gauge' in fig['data'][0]['mode'], "Indicator chart should have a gauge mode"