"""
bert_tiny_modified.py: This module provides a pipeline for training, evaluating, and deploying a BERT-based text classification model.
It includes functions for text preprocessing, model training, evaluation, and ONNX export, with visualization tools like classification reports and ROC-AUC curves.

Functions:
    clean_text(text): Preprocesses text by removing unwanted elements (URLs, mentions, special characters, etc.) and converting to lowercase.
    encode_and_save(texts, labels, tokenizer, max_len, save_path, batch_size=1024): Encodes texts and labels, then saves them as compressed NPZ files.
    load_npz(path): Loads encoded data from an NPZ file and returns PyTorch tensors.
    train_pipeline(texts, labels, model_name, max_len, batch_size, epochs, train_npz_path, val_npz_path, lr=2e-5, fast_run=False, use_profiler=False): Main pipeline for model training and evaluation, including data preprocessing and performance metrics calculation.
    evaluate_model(model, dataloader, label_encoder): Evaluates the model using classification metrics (precision, recall, F1-score, AUC-ROC) and generates a confusion matrix heatmap.
    export_onnx(model, tokenizer, save_path, max_len): Exports the trained model to ONNX format for efficient inference.
    onnx_predict(texts, tokenizer, label_encoder, onnx_path, max_len): Uses a pre-trained ONNX model for making predictions on new data.
    plot_roc_auc(model, dataloader, label_encoder): Plots ROC-AUC curves for model evaluation.

Classes:
    LightningTextClassifier: A subclass of PyTorch Lightning’s LightningModule for training and evaluating a BERT-based text classification model, with mixed-precision support and logging.

Usage:
    This module is designed for end-to-end text classification using BERT-based models. It includes data preprocessing, model training, evaluation, and deployment, with ONNX support for efficient inference.

Dependencies:
    - PyTorch
    - PyTorch Lightning
    - HuggingFace Transformers
    - scikit-learn
    - ONNX Runtime
    - Matplotlib
    - NumPy
    - pandas
    - joblib
"""

import os, re, gc, warnings, torch, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import onnxruntime as ort

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import torch.nn.functional as F
import joblib

# Suppress parallelism and warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

def clean_text(text):
    """
    Cleans the input text by performing several transformations:
    1. Removes URLs (both HTTP and WWW links).
    2. Replaces mentions of Twitter handles (@username) with a generic @user.
    3. Removes hashtags (#hashtag).
    4. Replaces multiple spaces with a single space.
    5. Removes non-alphanumeric characters (except spaces).
    6. Strips leading and trailing whitespace and converts the text to lowercase.

    Args:
        text (str): The input string that needs to be cleaned.

    Returns:
        str: The cleaned text, transformed according to the above rules.
    
    Example:
        clean_text("Check out @username's profile! Visit http://example.com #coolstuff")
        # Returns: "check out @user s profile visit user coolstuff"
    """
    # Remove URLs (http, https, and www)
    url_pattern = r'https?://\S+|www\.\S+'
    text_without_urls = re.sub(url_pattern, '', text)
    
    # Retain the @ symbol and remove numbers
    text = re.sub(r'@([a-zA-Z]+)\d*', r'@\1', text_without_urls)

    # Remove hashtags (#hashtag)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-alphanumeric characters (except spaces)
    text = re.sub(r'[^a-zA-Z0-9\s@]', '', text)
    return text.strip().lower()

class LightningTextClassifier(pl.LightningModule):
    """
    A PyTorch Lightning model for sequence classification using a pre-trained transformer model.
    This class handles training, validation, and optimizer configuration for text classification tasks.
    
    Args:
        model_name (str): The name of the pre-trained transformer model to be used for sequence classification.
        num_labels (int): The number of output labels/classes for the classification task.
        class_weights (torch.Tensor): The class weights to handle class imbalance in the loss function.
        lr (float): The learning rate for the optimizer.
        epochs (int): The number of training epochs.
        train_loader_len (int): The length of the training data loader used for calculating the total number of training steps.

    Attributes:
        model (transformers.PreTrainedModel): The pre-trained transformer model used for sequence classification.
        loss_fn (torch.nn.CrossEntropyLoss): The loss function with class weights applied.
        lr (float): The learning rate for the optimizer.
        epochs (int): The number of training epochs.
        train_loader_len (int): The length of the training data loader.
        val_preds (list): A list of validation predictions collected during the validation phase.
        val_probs (list): A list of validation probabilities collected during the validation phase.
        val_labels (list): A list of true labels for the validation set.
        train_accs (list): A list of training accuracies for each batch.

    Methods:
        forward(input_ids, attention_mask):
            Performs a forward pass through the model using input_ids and attention_mask.

        training_step(batch, batch_idx):
            Performs a single step of training, calculating loss and accuracy, and logs metrics.

        validation_step(batch, batch_idx):
            Performs a single step of validation, calculating predictions, probabilities, and accuracy.

        on_validation_epoch_end():
            Handles the end of the validation epoch, collecting predictions, probabilities, and labels.

        configure_optimizers():
            Configures the optimizer and learning rate scheduler for training.

    Example:
        # Initialize the model
        model = LightningTextClassifier(
            model_name="bert-base-uncased",
            num_labels=2,
            class_weights=torch.tensor([1.0, 2.0]),
            lr=5e-5,
            epochs=3,
            train_loader_len=1000
        )
        
        # Use the model in PyTorch Lightning Trainer
        trainer = pl.Trainer(max_epochs=3)
        trainer.fit(model, train_dataloader, val_dataloader)
    """
    
    def __init__(self, model_name, num_labels, class_weights, lr, epochs, train_loader_len):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.loss_fn = CrossEntropyLoss(weight=class_weights)
        self.lr = lr
        self.epochs = epochs
        self.train_loader_len = train_loader_len
        self.val_preds, self.val_probs, self.val_labels = [], [], []
        self.train_accs = []

    def forward(self, input_ids, attention_mask):
        """
        Performs a forward pass through the model.

        Args:
            input_ids (torch.Tensor): The input tensor containing token IDs for the sequence.
            attention_mask (torch.Tensor): The input tensor indicating which tokens are padding and should be ignored.

        Returns:
            transformers.ModelOutput: The output of the model containing logits for sequence classification.
        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        """
        Performs a single step of training, calculating loss and accuracy, and logs metrics.

        Args:
            batch (tuple): A tuple containing input_ids, attention_mask, and labels for the batch.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask)
        loss = self.loss_fn(outputs.logits, labels)
        preds = torch.argmax(outputs.logits, dim=1)
        acc = (preds == labels).float().mean()
        self.train_accs.append(acc.item())  # Append training accuracy for each batch
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a single step of validation, calculating predictions, probabilities, and accuracy.

        Args:
            batch (tuple): A tuple containing input_ids, attention_mask, and labels for the batch.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The computed accuracy for the batch.
        """
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask)
        probs = F.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        self.val_preds.append(preds.cpu())
        self.val_probs.append(probs.cpu())
        self.val_labels.append(labels.cpu())

        acc = (preds == labels).float().mean()
        self.log("val_acc", acc, prog_bar=True)
        return acc

    def on_validation_epoch_end(self):
        """
        Handles the end of the validation epoch, collecting predictions, probabilities, and labels.
        Clears the validation lists and stores the results for the entire epoch.
        """
        preds = torch.cat(self.val_preds).numpy()
        probs = torch.cat(self.val_probs).numpy()
        labels = torch.cat(self.val_labels).numpy()

        self.val_preds.clear()
        self.val_probs.clear()
        self.val_labels.clear()

        self.predictions_epoch = preds
        self.labels_epoch = labels
        self.probs_epoch = probs

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for training.

        Returns:
            tuple: A tuple containing the optimizer and scheduler configuration.
        """
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = get_scheduler(
            "linear", optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=self.epochs * self.train_loader_len
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]



def encode_and_save(texts, labels, tokenizer, max_len, save_path, batch_size=1024):
    """
    Encodes a list of texts into input IDs and attention masks using a tokenizer, and saves them as a compressed NPZ file.

    This function processes the input texts in batches, cleans them, tokenizes them, and saves the encoded input IDs, attention masks,
    and labels into a compressed `.npz` file at the specified save path. The function also ensures the directory structure exists 
    before saving.

    Args:
        texts (list of str): A list of raw text sequences to be encoded.
        labels (list of int): A list of integer labels corresponding to the texts.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding the text sequences.
        max_len (int): The maximum sequence length to pad or truncate the texts to.
        save_path (str): The path where the encoded data will be saved (in `.npz` format).
        batch_size (int, optional): The batch size to use when processing the texts (default is 1024).

    Returns:
        tuple: A tuple containing:
            - input_ids (torch.Tensor): The tensor of encoded input IDs.
            - attention_mask (torch.Tensor): The tensor of attention masks.
            - labels_tensor (torch.Tensor): The tensor of labels corresponding to the encoded texts.
    
    Example:
        texts = ["This is an example.", "Another example sentence."]
        labels = [0, 1]
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        encode_and_save(texts, labels, tokenizer, max_len=128, save_path="data/encoded_data.npz")
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    input_ids_all, attention_mask_all = [], []
    for i in range(0, len(texts), batch_size):
        batch = [clean_text(t) for t in texts[i:i+batch_size]]
        enc = tokenizer(batch, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
        input_ids_all.append(enc["input_ids"])
        attention_mask_all.append(enc["attention_mask"])
    input_ids = torch.cat(input_ids_all)
    attention_mask = torch.cat(attention_mask_all)
    labels_tensor = torch.tensor(labels[:len(input_ids)], dtype=torch.long)
    np.savez_compressed(save_path, input_ids=input_ids.numpy(), attention_masks=attention_mask.numpy(), labels=labels_tensor.numpy())
    return input_ids, attention_mask, labels_tensor

def load_npz(path):
    """
    Loads a compressed `.npz` file containing encoded input IDs, attention masks, and labels.

    This function loads the encoded text data (input IDs, attention masks) and labels from a previously saved compressed `.npz` file
    and returns them as tensors.

    Args:
        path (str): The file path to the compressed `.npz` file containing the encoded data.

    Returns:
        tuple: A tuple containing:
            - input_ids (torch.Tensor): The tensor of input IDs.
            - attention_mask (torch.Tensor): The tensor of attention masks.
            - labels (torch.Tensor): The tensor of labels.

    Example:
        input_ids, attention_mask, labels = load_npz("data/encoded_data.npz")
    """
    data = np.load(path)
    return torch.tensor(data["input_ids"]), torch.tensor(data["attention_masks"]), torch.tensor(data["labels"])


def train_pipeline(
    texts, labels, model_name, max_len, batch_size,
    epochs, train_npz_path, val_npz_path,
    lr=2e-5, fast_run=False, use_profiler=False
):
    """
    A pipeline for training a text classification model using a pre-trained transformer (e.g., BERT) and PyTorch Lightning.

    This function prepares the data by encoding the text sequences, handles class imbalance using oversampling, and then trains 
    a model using a specified number of epochs and learning rate. The model is validated during training, and the final evaluation 
    is performed on the validation set. The tokenizer and label encoder are saved for future use in inference.

    Args:
        texts (list of str): A list of raw text sequences to be classified.
        labels (list of int): A list of integer labels corresponding to the texts.
        model_name (str): The name of the pre-trained transformer model to use (e.g., 'bert-base-uncased').
        max_len (int): The maximum sequence length to pad or truncate the texts to.
        batch_size (int): The batch size used for training and validation.
        epochs (int): The number of training epochs.
        train_npz_path (str): The path to the cached `.npz` file for training data.
        val_npz_path (str): The path to the cached `.npz` file for validation data.
        lr (float, optional): The learning rate for the optimizer (default is 2e-5).
        fast_run (bool, optional): If True, performs a quick run with a subset of the data (default is False).
        use_profiler (bool, optional): If True, enables profiling to analyze training performance (default is False).

    Returns:
        tuple: A tuple containing:
            - model (LightningTextClassifier): The trained PyTorch Lightning model.
            - tokenizer (PreTrainedTokenizer): The tokenizer used for encoding the text sequences.
            - label_encoder (LabelEncoder): The label encoder used to encode labels.

    Example:
        model, tokenizer, label_encoder = train_pipeline(
            texts=train_texts, labels=train_labels, model_name="bert-base-uncased",
            max_len=128, batch_size=32, epochs=3, train_npz_path="train_data.npz",
            val_npz_path="val_data.npz"
        )
    """
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels_encoded, test_size=0.2, stratify=labels_encoded, random_state=42
    )

    # Handle imbalance in the training data (only apply to training set)
    df_train = pd.DataFrame({"text": [clean_text(t) for t in X_train], "label": y_train})
    X_resampled, y_resampled = RandomOverSampler(random_state=42).fit_resample(df_train[["text"]], df_train["label"])
    df_train_resampled = pd.DataFrame({"text": X_resampled["text"], "label": y_resampled})

    # Now encode and save the training and validation data
    if not os.path.exists(train_npz_path):
        print("Encoding and saving training data...")
        input_ids, attention_masks, labels_tensor = encode_and_save(df_train_resampled["text"].tolist(), y_resampled.tolist(), tokenizer, max_len, train_npz_path)
    else:
        print("Loading cached training data...")
        input_ids, attention_masks, labels_tensor = load_npz(train_npz_path)

    if not os.path.exists(val_npz_path):
        print("Encoding and saving validation data...")
        val_ids, val_masks, val_labels = encode_and_save(X_val.tolist(), y_val.tolist(), tokenizer, max_len, val_npz_path)
    else:
        print("Loading cached validation data...")
        val_ids, val_masks, val_labels = load_npz(val_npz_path)

    # Build TensorDataset
    train_dataset = TensorDataset(input_ids, attention_masks, labels_tensor)
    val_dataset = TensorDataset(val_ids, val_masks, val_labels)

    print("After TensorDataset")
    # Compute weights
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    class_weights = torch.tensor(
        compute_class_weight("balanced", classes=np.unique(y_train), y=y_train),
        dtype=torch.float
    ).to(device)

    model = LightningTextClassifier(
        model_name, num_labels=len(label_encoder.classes_),
        class_weights=class_weights, lr=lr,
        epochs=epochs, train_loader_len=len(train_dataset) // batch_size
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print("After DataLoader")
    # Clean memory
    torch.cuda.empty_cache()
    gc.collect()

    print("Before Training")
    trainer = pl.Trainer(
        max_epochs=epochs,
        precision=16,  # Mixed precision
        accelerator="gpu" if torch.cuda.is_available() else "mps",
        devices=1,
        callbacks=[
            EarlyStopping(monitor="val_acc", mode="max", patience=2),
            ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1)
        ],
        log_every_n_steps=10,
        fast_dev_run=fast_run,
        profiler="simple" if use_profiler else None  # Optional profiling
    )

    trainer.fit(model, train_loader, val_loader)

    print(f"Training Accuracy: {np.mean(model.train_accs):.4f}")

    print("\n Running final evaluation on validation set...")
    evaluate_model(model.model, val_loader, label_encoder)

    plot_roc_auc(model, val_loader, label_encoder)

    # Save tokenizer and label encoder for prediction use
    tokenizer.save_pretrained("model/bert_tiny_tokenizer")
    joblib.dump(label_encoder, "model/bert_tiny_label_encoder.pkl")
    print("Tokenizer and label encoder saved.")

    return model, tokenizer, label_encoder

def evaluate_model(model, dataloader, label_encoder):
    """
    Evaluates the performance of a trained model on a given dataloader and prints various metrics.

    This function performs evaluation on the model by calculating classification metrics, confusion matrix, 
    per-class accuracy, overall accuracy, and AUC-ROC (both binary and multi-class if applicable). It also 
    generates a confusion matrix heatmap for visual inspection.

    Args:
        model (torch.nn.Module): The trained PyTorch model to evaluate.
        dataloader (torch.utils.data.DataLoader): The DataLoader containing the validation or test dataset.
        label_encoder (sklearn.preprocessing.LabelEncoder): The label encoder used to decode the class labels for display.

    Prints:
        - Classification report including precision, recall, f1-score, and support for each class.
        - Per-class accuracy.
        - Overall accuracy.
        - AUC-ROC score (both for binary and multi-class classification).
        - Confusion matrix heatmap.

    Example:
        evaluate_model(model, val_loader, label_encoder)
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    print("\n Classification Report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    print("\n Per-Class Accuracy:")
    for i, cls in enumerate(label_encoder.classes_):
        print(f"{cls}: {per_class_acc[i]:.4f}")

    overall_accuracy = np.mean(y_pred == y_true)
    print(f"Overall Validation Accuracy: {overall_accuracy:.4f}")

    if y_prob.shape[1] > 2:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
            print(f"\n AUC-ROC (OvR): {auc:.4f}")
        except:
            print(" AUC-ROC could not be computed (check label or output shape)")
    else:
        try:
            auc = roc_auc_score(y_true, y_prob[:, 1])
            print(f"\n AUC-ROC (Binary): {auc:.4f}")
        except:
            print(" Binary AUC-ROC could not be computed")
    
    # Confusion Matrix Heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(" Confusion Matrix Heatmap")
    plt.tight_layout()
    plt.show()

def export_onnx(model, tokenizer, save_path, max_len):
    """
    Export a trained model to the ONNX format for inference.

    This function converts a PyTorch model into the ONNX format and saves it to a specified file path.
    The exported model can be loaded and used for inference in environments that support ONNX (e.g., ONNX Runtime).

    Args:
        model (torch.nn.Module): The trained model to be exported to ONNX format.
        tokenizer (PreTrainedTokenizer): The tokenizer used to encode the input text.
        save_path (str): The path where the ONNX model will be saved.
        max_len (int): The maximum sequence length for padding/truncating the input text.

    Example:
        export_onnx(model, tokenizer, "model.onnx", max_len=128)
    """
    print("Exporting to ONNX...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dummy_input = tokenizer.encode_plus(
        "This is a dummy input for ONNX export.", return_tensors="pt",
        max_length=max_len, padding="max_length", truncation=True
    )
    model.model.to("cpu").eval()
    torch.onnx.export(
        model.model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        save_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size"}, "attention_mask": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=14
    )
    print(f"Model saved to {save_path}")

def onnx_predict(texts, tokenizer, label_encoder, onnx_path, max_len):
    """
    Perform inference using an ONNX model.

    This function takes a list of input texts, tokenizes and processes them, and runs inference with the ONNX model
    specified by the given path. The results are returned as decoded class labels.

    Args:
        texts (list of str): A list of texts to make predictions for.
        tokenizer (PreTrainedTokenizer): The tokenizer used to encode the input text.
        label_encoder (LabelEncoder): The label encoder used to decode the predicted labels.
        onnx_path (str): The path to the saved ONNX model.
        max_len (int): The maximum sequence length for padding/truncating the input text.

    Returns:
        numpy.ndarray: The predicted labels decoded from the label encoder.

    Example:
        predictions = onnx_predict(["This is a test sentence."], tokenizer, label_encoder, "model.onnx", max_len=128)
    """
    print("ONNX prediction...")
    session = ort.InferenceSession(onnx_path)
    cleaned_texts = [clean_text(t) for t in texts]
    encodings = tokenizer(
        cleaned_texts, return_tensors="np", max_length=max_len,
        padding="max_length", truncation=True
    )
    input_ids_np = encodings["input_ids"].astype(np.int64)
    attention_masks_np = encodings["attention_mask"].astype(np.int64)

    logits = session.run(["logits"], {
        "input_ids": input_ids_np,
        "attention_mask": attention_masks_np
    })[0]

    preds = np.argmax(logits, axis=1)
    return label_encoder.inverse_transform(preds)

def plot_roc_auc(model, dataloader, label_encoder):
    """
    Plot the ROC curve and compute the AUC-ROC for each class.

    This function evaluates the model's performance on a given dataloader by calculating the ROC curve and 
    AUC-ROC for each class. It then plots the ROC curves for each class and displays the corresponding AUC-ROC values.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        dataloader (torch.utils.data.DataLoader): The DataLoader containing the dataset for evaluation.
        label_encoder (sklearn.preprocessing.LabelEncoder): The label encoder used to decode the class labels.

    Returns:
        None: This function only generates a plot of the ROC curves for each class.

    Example:
        plot_roc_auc(model, val_loader, label_encoder)
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)

    # ROC-AUC for each class
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(len(label_encoder.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot the ROC curve
    plt.figure(figsize=(10, 7))
    for i in range(len(label_encoder.classes_)):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{label_encoder.classes_[i]} (AUC = {roc_auc[i]:.4f})')

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (per Class)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
