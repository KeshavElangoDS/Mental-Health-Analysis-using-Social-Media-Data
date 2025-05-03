import pytest
from mentalhealth.bert_tiny_modified import clean_text
import pytest
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from mentalhealth.bert_tiny_modified import (
    LightningTextClassifier, encode_and_save,
    load_npz, plot_roc_auc
)
import matplotlib as plt
from unittest.mock import patch

# Use non-interactive backend for tests
plt.use("Agg")

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

# Configuration
MODEL_NAME = "prajjwal1/bert-tiny"
MAX_LEN = 128
BATCH_SIZE = 64
EPOCHS = 5

def test_clean_text_removes_urls():
    text = "Check this out http://example.com"
    cleaned = clean_text(text)
    print(f"Cleaned text: {cleaned}")
    assert cleaned == "check this out"

def test_clean_text_replaces_twitter_handles():
    text = "Follow me on Twitter @user123"
    cleaned = clean_text(text)
    assert cleaned == "follow me on twitter @user"

def test_clean_text_removes_hashtags():
    text = "This is a #cool hashtag #example"
    cleaned = clean_text(text)
    assert cleaned == "this is a hashtag"

def test_clean_text_replaces_multiple_spaces():
    text = "This   has    multiple    spaces"
    cleaned = clean_text(text)
    assert cleaned == "this has multiple spaces"

def test_clean_text_removes_non_alphanumeric_characters():
    text = "Hello! How are you? :)"
    cleaned = clean_text(text)
    assert cleaned == "hello how are you"

def test_clean_text_trims_whitespace_and_lowercase():
    text = "   Lots of leading and trailing spaces   "
    cleaned = clean_text(text)
    assert cleaned == "lots of leading and trailing spaces"

def test_clean_text_combined_case():
    text = "Visit my profile at http://myprofile.com and follow @user1 for cool #content"
    cleaned = clean_text(text)
    assert cleaned == "visit my profile at and follow @user for cool"

@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)

@pytest.fixture
def dummy_data():
    texts = ["I'm feeling great!", "Terrible day...", "Okay, I guess", "Very anxious", "All is well"]
    labels = [1, 0, 1, 0, 1]
    return texts, labels

def test_encode_and_save_and_load(tmp_path, tokenizer, dummy_data):
    texts, labels = dummy_data
    save_path = tmp_path / "encoded_data.npz"
    
    input_ids, attention_mask, labels_tensor = encode_and_save(
        texts, labels, tokenizer, max_len=MAX_LEN, save_path=str(save_path), batch_size=BATCH_SIZE
    )

    # Load the saved data
    input_ids_loaded, attn_mask_loaded, labels_loaded = load_npz(str(save_path))

    assert input_ids.shape == input_ids_loaded.shape
    assert torch.equal(input_ids, input_ids_loaded)
    assert torch.equal(attention_mask, attn_mask_loaded)
    assert torch.equal(labels_tensor, labels_loaded)

def test_model_forward(tokenizer):
    model = LightningTextClassifier(
        model_name=MODEL_NAME,
        num_labels=2,
        class_weights=torch.tensor([1.0, 1.0]),
        lr=2e-5,
        epochs=EPOCHS,
        train_loader_len=10
    )
    
    sample_texts = ["This is a test", "Another input"]
    encodings = tokenizer(sample_texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    outputs = model(encodings['input_ids'], encodings['attention_mask'])

    assert outputs.logits.shape == (2, 2)

def test_training_step_with_trainer(tokenizer):
    model = LightningTextClassifier(
        model_name="prajjwal1/bert-tiny",
        num_labels=2,
        class_weights=torch.tensor([1.0, 1.0]),
        lr=2e-5,
        epochs=1,
        train_loader_len=1
    )

    texts = ["Good day", "Bad mood"]
    labels = torch.tensor([1, 0])
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], labels)
    loader = DataLoader(dataset, batch_size=2)

    trainer = Trainer(max_epochs=1, enable_model_summary=False, logger=False, enable_checkpointing=False)
    trainer.fit(model, loader, loader)

@pytest.fixture
def dummy_eval_data(tokenizer):
    texts = ["happy", "sad", "content", "angry"]
    labels = [1, 0, 1, 0]
    le = LabelEncoder()
    labels_enc = le.fit_transform(labels)

    enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    dataset = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(labels_enc))
    dataloader = DataLoader(dataset, batch_size=2)
    return dataloader, le


def test_plot_roc_auc_runs_without_error(tokenizer, dummy_eval_data):
    dataloader, label_encoder = dummy_eval_data

    model = LightningTextClassifier(
        model_name=MODEL_NAME,
        num_labels=2,
        class_weights=torch.tensor([1.0, 1.0]),
        lr=2e-5,
        epochs=1,
        train_loader_len=1
    )
    model.eval()

    with patch("matplotlib.pyplot.show"):  # prevent actual plot display
        plot_roc_auc(model, dataloader, label_encoder)

