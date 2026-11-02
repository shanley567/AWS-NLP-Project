
# AWS NLP Project

A containerized Natural Language Processing (NLP) project built using Python, Docker, TensorFlow Hub, and KaggleHub.

The project explores text embeddings, sentiment analysis, data preparation, and machine learning workflows using real-world datasets. Development is performed locally inside Docker, providing a foundation for future deployment to AWS services such as ECS Fargate, ECR, S3, and EFS.

---

# Project Structure

```text
AWS-NLP-Project
│
├── data/
│   ├── Twitter_Data.csv
│   ├── Spam_SMS.csv
│   └── ...
│
├── kaggle-cache/
│   └── Downloaded embedding models
│
├── output/
│   └── plot.png
│
├── src/
│   ├── main.py
│   ├── section1_embeddings.py
│   └── section2_data_prep.py
│
├── Dockerfile
├── compose.yaml
├── requirements.txt
└── README.md
```

---

# Features

## Section 1 - Word Embeddings

- Download and cache TensorFlow Hub embeddings using KaggleHub
- Generate 250-dimensional Wiki-Words embeddings
- Create sentence embeddings
- Calculate cosine similarity between words
- Perform PCA dimensionality reduction
- Generate visualization plots

## Section 2 - Data Preparation

- Load sentiment analysis datasets
- Clean missing values
- Standardize column names
- Validate labels
- Shuffle datasets
- Generate embeddings from text data
- Determine maximum sequence length for future RNN/LSTM models

---

# Technologies

- Python 3.11
- Docker
- Docker Compose
- TensorFlow
- TensorFlow Hub
- KaggleHub
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

# Dataset

Current development uses:

```text
Twitter_Data.csv
```

Dataset schema:

| Column | Description     |
| ------ | --------------- |
| text   | Tweet text      |
| label  | Sentiment class |

Sentiment labels:

```text
-1 = Negative
 0 = Neutral
 1 = Positive
```

After preprocessing:

```text
Rows: 162,969
```

Class distribution:

```text
Positive : 72,249
Neutral  : 55,211
Negative : 35,509
```

---

# Model

Current embedding model:

```text
google/wiki-words/tensorFlow2/250
```

Features:

- 250-dimensional word embeddings
- Cached locally using KaggleHub
- Automatically reused between container executions

Model cache location:

```text
kaggle-cache/
```

Docker volume mapping:

```yaml
- ./kaggle-cache:/root/.cache/kagglehub
```

This prevents repeated model downloads and significantly reduces startup time.

---

# Docker Development Workflow

## Build

```powershell
docker compose up --build
```

Required when:

- Dockerfile changes
- requirements.txt changes

---

## Run

```powershell
docker compose up
```

Used for normal development.

Source code is mounted directly into the container:

```yaml
- ./src:/app/src
```

allowing code changes without rebuilding the image.

---

# Processing Flow

```text
Dataset
    │
    ▼
Data Preparation
    │
    ▼
Text Cleaning
    │
    ▼
Label Validation
    │
    ▼
Word Embeddings
    │
    ▼
Sentence Embeddings
    │
    ▼
Dimensionality Reduction
    │
    ▼
Visualization
```

---

# Current Status

Completed:

- [X] Docker containerization
- [X] TensorFlow Hub integration
- [X] KaggleHub model caching
- [X] Word embeddings
- [X] Sentence embeddings
- [X] Similarity calculations
- [X] PCA visualization
- [X] Dataset loading
- [X] Dataset preprocessing
- [X] Dataset validation

In Progress:

- [ ] Train/Test Split
- [ ] Sequence Padding
- [ ] RNN / LSTM Model Development
- [ ] Classification Training
- [ ] Performance Evaluation

Future Enhancements:

- [ ] AWS ECR integration
- [ ] ECS Fargate deployment
- [ ] S3 data storage
- [ ] Automated CI/CD pipeline
- [ ] Infrastructure as Code using Terraform

---

# Running the Project

Start the application:

```powershell
docker compose up
```

Example output:

```text
[INFO] - Loading dataset: /data/Twitter_Data.csv
[INFO] - Dataset shape after cleaning: (162969, 2)
[INFO] - Labels detected: [-1, 0, 1]

[INFO] - Loading embedding model...
[INFO] - Embedding model loaded.

[INFO] - Processed 900 rows

[INFO] Generated embeddings for 1000 texts
```

---

# Learning Objectives

This project demonstrates:

- NLP preprocessing pipelines
- Word embeddings and vectorization
- Containerized machine learning workflows
- Model caching strategies
- Dataset validation and cleaning
- Reproducible development environments
- Cloud-ready architecture patterns
