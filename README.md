# Multi-Document News Summarization 📰🔍

## 🌟 Project Overview

This research project investigates automatic multi-document summarization techniques, focusing on combining extractive and abstractive approaches to generate high-quality, concise summaries from multiple news articles.

### 🎯 Research Question
Can integrating extractive and abstractive approaches improve summary quality compared to individual methods?

## 📋 Key Objectives

- Explore the potential benefits of combining extractive and abstractive summarization approaches
- Develop a multi-document summarization system capable of generating informative summaries from sets of journalistic articles

## 🧠 Methodology

### 📊 Dataset
- **Multi-News Dataset**: 
  - 56,216 news article and summary pairs
  - Sources from over 1,500 news outlets
  - Significant document length variability (median ~1,540 words)

### 🔬 Approach
A hybrid model combining two key components:

#### 1. Extractive Module
Two techniques explored:
- **TextRank**: Graph-based algorithm identifying key sentences
- **K-means Clustering**: Segmenting sentences into clusters and selecting representative phrases

#### 2. Abstractive Module
Two transformer-based models evaluated:
- **Flan-T5-small**: Lightweight, flexible text-to-text model
- **Distill-BART**: Efficient text generation model with bidirectional encoding

## 📏 Evaluation Metrics
- ROUGE-1, ROUGE-2, ROUGE-L
- BERTScore-F1

## 🚀 Proposed Future Improvements
- Fine-tuning models on specific multi-document datasets
- Implementing hierarchical summarization approach
- Developing more sophisticated integration of extractive and abstractive techniques

## 💻 Technologies Used
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-FF6F61?style=for-the-badge)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-4CAF50?style=for-the-badge)

- Python
- Transformer models (T5, BART)
- Machine learning libraries (Hugging Face, pandas)
- Evaluation metrics (ROUGE, BERTScore)
