"""
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
import joblib

# Suppress parallelism and warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip().lower()

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import torch.nn.functional as F

class LightningTextClassifier(pl.LightningModule):
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
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
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
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = get_scheduler(
            "linear", optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=self.epochs * self.train_loader_len
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]



def encode_and_save(texts, labels, tokenizer, max_len, save_path, batch_size=1024):
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
    data = np.load(path)
    return torch.tensor(data["input_ids"]), torch.tensor(data["attention_masks"]), torch.tensor(data["labels"])


def train_pipeline(
    texts, labels, model_name, max_len, batch_size,
    epochs, train_npz_path, val_npz_path,
    lr=2e-5, fast_run=False, use_profiler=False
):
    
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
