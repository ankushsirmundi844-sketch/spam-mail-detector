# 🛡️ Spam Mail Detector
> End-to-end NLP classification pipeline · SMS spam vs ham · Streamlit web app

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

A complete machine learning pipeline that classifies SMS messages as **spam** or **ham** (legitimate), built using classical NLP and sklearn. Features an interactive Streamlit web app for live predictions and batch classification.

**Dataset:** SMS Spam Collection (UCI ML Repository) — 5,574 messages, 13.4% spam

---

## 🔧 Pipeline Stages

```
Raw SMS Text
     │
     ▼
┌─────────────────────────────┐
│  Text Preprocessing         │
│  • Lowercase                │
│  • URL / phone tokenisation │
│  • Currency token           │
│  • Punctuation removal      │
│  • Stopword filtering       │
└────────────┬────────────────┘
             │
     ┌───────┴────────┐
     ▼                ▼
TF-IDF Bigrams    14 Hand-crafted
(8,000 features)  Features
     │                │
     └───────┬────────┘
             ▼
     ┌───────────────┐
     │  Classifiers  │
     │  • Naive Bayes│
     │  • Complement │
     │  • Log. Reg.  │
     │  • LinearSVC  │
     │  • Ensemble   │
     └───────┬───────┘
             ▼
     GridSearchCV Tuning
             │
             ▼
     Best Model → pickle
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/spam-detector.git
cd spam-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate/place dataset
```bash
# Option A: Use the provided generator
python data/generate_dataset.py

# Option B: Use real UCI dataset
# Download from https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
# Place as data/sms.tsv
```

### 4. Train the model
```bash
python train.py
```

### 5. Launch the web app
```bash
streamlit run app.py
```

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| Naive Bayes (BoW) | 98.6% | 97.8% | 96.4% | 97.1% | 0.994 |
| Complement NB | 98.9% | 98.3% | 97.1% | 97.7% | 0.997 |
| Logistic Regression | 99.2% | 99.1% | 97.8% | 98.4% | 0.999 |
| Linear SVM | 99.0% | 98.8% | 97.5% | 98.1% | — |
| **LR (Tuned)** | **99.4%** | **99.3%** | **98.2%** | **98.7%** | **0.999** |
| Ensemble (Voting) | 99.3% | 99.1% | 98.0% | 98.5% | 0.999 |

> Results on real UCI SMS Spam Collection dataset (20% test split)

---

## 🖥️ Web App Features

| Page | Description |
|------|-------------|
| **Live Classifier** | Type/paste a message → instant prediction with confidence score and signal breakdown |
| **Analytics Dashboard** | EDA charts, word clouds, feature correlation matrix, length distributions |
| **Batch Test** | Classify multiple messages at once, download results as CSV |
| **About** | Project structure, pipeline overview, skills summary |

---

## 📁 Project Structure

```
spam_detector/
├── data/
│   ├── sms.csv                  # Dataset (or sms.tsv from UCI)
│   └── generate_dataset.py      # Synthetic dataset generator
├── models/
│   └── best_model.pkl           # Saved best model (after train.py)
├── outputs/
│   └── results_dashboard.png    # Training visualisation (8 plots)
├── preprocessor.py              # Text cleaning + feature engineering
├── train.py                     # Full training pipeline
├── app.py                       # Streamlit web application
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🛠️ Skills Demonstrated

- **Text Preprocessing:** regex, URL/phone/currency tokenisation, stopword removal
- **Feature Extraction:** Bag of Words, TF-IDF with bigrams, hand-crafted NLP features
- **Model Training:** Naive Bayes, Complement NB, Logistic Regression, LinearSVC
- **Hyperparameter Tuning:** GridSearchCV (3-fold CV, F1-optimised)
- **Ensemble Methods:** Soft Voting Classifier
- **Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC, 5-fold Cross-Validation
- **Visualisation:** Confusion matrix, ROC curves, word clouds, feature importance
- **Deployment:** Streamlit interactive web app with batch prediction and CSV export
- **Model Persistence:** pickle serialisation

---

## 📦 Requirements

```
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
wordcloud>=1.9
streamlit>=1.30
scipy>=1.11
```

---

## 📄 License
MIT — free to use for academic and personal projects.
