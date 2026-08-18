# BERT Clickbait Detection

A machine learning project that uses **BERT (Bidirectional Encoder Representations from Transformers)** to detect clickbait headlines in news articles. This project leverages state-of-the-art NLP techniques to classify whether a headline is clickbait or not.

## 📋 Project Overview

This project implements a binary classification model using BERT to identify clickbait headlines. The model is fine-tuned on a clickbait dataset and can distinguish between legitimate headlines and clickbait content with high accuracy.

### What is BERT?

**BERT** (Bidirectional Encoder Representations from Transformers) is a pre-trained language model developed by Google that:
- Uses a **transformer architecture** with bidirectional training
- Understands context from both directions (left and right) in text
- Has been pre-trained on massive amounts of unlabeled text data
- Achieves state-of-the-art results on various NLP tasks including text classification
- Uses WordPiece tokenization for efficient text processing

In this project, we use **bert-base-uncased** which has 12 layers, 768 hidden units, and 12 attention heads.

## 🎯 What is Clickbait?

Clickbait refers to headlines designed to attract attention and entice users to click on them, often through:
- Exaggeration or sensationalism
- Misleading information
- Emotional manipulation
- Unresolved curiosity ("You won't believe what happened next...")
- Shock value

This project aims to automatically detect such manipulative headlines.

## 🚀 Features

- **Pre-trained BERT Model**: Uses `bert-base-uncased` from Hugging Face
- **Binary Classification**: Classifies headlines as Clickbait (1) or Not Clickbait (0)
- **Efficient Tokenization**: Automatic padding and truncation using BERT tokenizer
- **Fine-tuning**: Custom training loop with validation and early stopping
- **Performance Metrics**: Accuracy and F1-Score evaluation
- **Google Colab Compatible**: Runs seamlessly on Google Colab with GPU support

## 📦 Dependencies
