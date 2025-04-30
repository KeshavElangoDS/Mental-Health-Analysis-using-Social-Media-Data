# Mental Health Analysis using Social Media Data

## Problem Statement

This project applies machine learning and NLP techniques to identify and classify social media posts related to mental health issues. Platforms like Twitter and Reddit have become spaces where users discuss personal challenges, making it essential to detect such content for timely intervention and support.

The objective is to develop models that detect mental health-related posts and categorize them into specific conditions such as **anxiety**, **depression**, **ADHD**, **PTSD**, and others.

---

## Key Steps

### 1. Data Collection
- Data sources include Reddit (PushShift API), and public datasets from MIT Libraries.
- Focused subreddits: `r/depression`, `r/Anxiety`, `r/bipolar`, `r/BPD`, `r/schizophrenia`, `r/autism`.
- Keywords: `anxiety`, `depression`, `stress`, `suicide`, and related terms.

### 2. Data Preprocessing
- Cleaned by removing non-ASCII characters, digits, and excess spaces.
- Lemmatization using **spaCy**.
- Tokenization and text storage using **TF-IDF**.
- Efficient processing via **parallelization** (`ProcessPoolExecutor`).

### 3. Feature Engineering
- **TF-IDF** (CountVectorizer, TfidfVectorizer) for traditional ML models.
- **BERT tokenizer** used for transformer-based modeling, saved in `.npz` format.

### 4. Model Training
- **Traditional ML Models**:
  - Logistic Regression, Multinomial Naive Bayes, LightGBM, XGBoost.
  - Exported in **ONNX** format.
- **Transformer Model**:
  - BERT fine-tuned using **PyTorch Lightning**.
  - Class imbalance addressed via `RandomOverSampler` after train-test split.
  - Training monitored with **early stopping** (patience = 2).

---

## Performance Metrics

### Accuracy
- **Training Accuracy**: `0.8992`
- **Validation Accuracy**: `0.9589`
- **AUC-ROC (OvR)**: `0.9979`

### Classification Report (Validation Set – BERT)

| Class             | Precision | Recall | F1-Score | Support  |
|------------------|-----------|--------|----------|----------|
| depression        | 0.85      | 0.63   | 0.72     | 126,898  |
| anxiety           | 0.92      | 0.96   | 0.94     | 126,898  |
| suicidewatch      | 0.79      | 0.93   | 0.85     | 126,898  |
| mentalhealth      | 0.89      | 0.97   | 0.93     | 126,898  |
| non_mental_health | 0.98      | 0.78   | 0.87     | 126,898  |
| Other Classes     | ~0.96–1.00| ~1.00  | ~0.99–1.00| ~126,898 |

**Total Samples Evaluated**: `2,284,161`  
**Overall Validation Accuracy**: `0.9589`  
**Challenging Classes**: `anxiety`, `depression`, and `mentalhealth` — high overlap causes confusion.

### Per-Class Accuracy Highlights
- depression: `0.6282`
- suicidewatch: `0.9256`
- non_mental_health: `0.7837`
- Most others: `> 0.99`

---

## Model Comparison
- **Best AUC-ROC (Traditional Models)**: XGBoost
- **Best Overall (Accuracy + ROC)**: BERT

---

## Data Sources
- MIT Libraries Dataset:
  - [Record at MIT Libraries](https://rdi.libraries.mit.edu/record/zenodo:3941387)
  - [Zenodo Mirror](https://zenodo.org/records/3941387)

---

## Dataset Citation

Low, D. M., Rumker, L., Torous, J., Cecchi, G., Ghosh, S. S., & Talkar, T. (2020).  
*Natural Language Processing Reveals Vulnerable Mental Health Support Groups and Heightened Health Anxiety on Reddit During COVID-19: Observational Study*.  
Journal of Medical Internet Research, 22(10), e22635.  
[Link to Article](https://www.jmir.org/2020/10/e22635)

BibTeX:
```bibtex
@article{low2020natural,
  title={Natural Language Processing Reveals Vulnerable Mental Health Support Groups and Heightened Health Anxiety on Reddit During COVID-19: Observational Study},
  author={Low, Daniel M and Rumker, Laurie and Torous, John and Cecchi, Guillermo and Ghosh, Satrajit S and Talkar, Tanya},
  journal={Journal of medical Internet research},
  volume={22},
  number={10},
  pages={e22635},
  year={2020},
  publisher={JMIR Publications Inc., Toronto, Canada}
}
```

---

## Tools and Technologies

- **NLP**: spaCy, NLTK, Hugging Face Transformers
- **ML/DL**: Scikit-learn, LightGBM, XGBoost, PyTorch Lightning, BERT
- **Data Handling**: pandas, NumPy, Parquet, `.npz`
- **Tokenization**: BERT tokenizer
- **Evaluation**: scikit-learn metrics, ROC AUC, per-class accuracy
- **Deployment**: ONNX model exports
- **Performance Optimization**: ProcessPoolExecutor, early stopping, `RandomOverSampler` (imblearn)

---

## Future Enhancements

- Integrate real-time analysis from Twitter and Reddit.
- Build interactive dashboards for public health monitoring.
- Incorporate explainability tools like **SHAP** or **LIME**.
- Explore additional transformer architectures (e.g., **RoBERTa**, **DistilBERT**).
