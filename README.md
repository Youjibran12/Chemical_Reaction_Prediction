---
title: ChemRxn Deep Learning
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: false
---



# Chemical Reaction Prediction
### Hybrid GRU + Transformer Ensemble for Chemical Reaction Prediction

ChemFusion is a deep learning framework for predicting chemical reaction products from reactant SMILES strings using a hybrid ensemble architecture combining a GRU-based Seq2Seq model and a Transformer model.

The system leverages chemically-aware tokenization, beam search decoding, attention mechanisms, and log-probability fusion to improve prediction accuracy and robustness for forward chemical reaction prediction tasks.

---

## Features

- Hybrid Ensemble Architecture (GRU + Transformer)
- Chemically-aware SMILES tokenization
- Attention-based Seq2Seq learning
- Transformer encoder-decoder architecture
- Beam Search Decoding
- Top-K prediction support
- Ensemble log-probability fusion
- Gradio-based interactive web interface
- Evaluation using BLEU, Levenshtein Accuracy, Character Accuracy, and Exact Match
- 3D Molecular Visualization
- Retrosynthesis support
- Batch prediction support
- CSV upload and evaluation

---

## System Architecture

The framework consists of:

1. SMILES Tokenizer
2. Vocabulary Builder
3. GRU Seq2Seq with Bahdanau Attention
4. Transformer Encoder-Decoder
5. Ensemble Fusion Module
6. Beam Search Decoder
7. Evaluation Pipeline
8. Gradio Deployment Interface

---

## Tech Stack

### Languages
- Python

### Deep Learning Frameworks
- PyTorch
- Hugging Face Transformers

### Frontend / Deployment
- Gradio

### Data Processing
- NumPy
- Pandas
- Regex
- RDKit

### Visualization
- Matplotlib
- Seaborn

### Environment
- Google Colab
- Jupyter Notebook

---

## Models Used

### 1. GRU Seq2Seq Model
- Bidirectional GRU Encoder
- Bahdanau Attention
- Autoregressive Decoder

### 2. Transformer Model
- Multi-Head Self Attention
- Positional Encoding
- Encoder-Decoder Architecture

### 3. Ensemble Strategy
Predictions from both models are combined using weighted log-probability fusion during beam search decoding.

---

## Dataset

Dataset Used:
- SLM4CRP with RTs benchmark dataset

Input:
- Reactant + Reagent SMILES

Output:
- Product SMILES

---

## Evaluation Metrics

The project evaluates performance using:

- Character-Level Accuracy
- BLEU Score
- Levenshtein Accuracy
- Exact Match Accuracy
- Prefix Match
- Top-K Accuracy

---

## Project Structure

```bash
ChemFusion/
│
├── models/
├── tokenizer/
├── training/
├── evaluation/
├── gradio_app/
├── notebooks/
├── saved_models/
├── utils/
├── results/
├── README.md
└── requirements.txt