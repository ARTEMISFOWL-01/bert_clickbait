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

```
transformers==4.55.2
datasets==4.0.0
torch==2.6.0
scikit-learn==1.6.1
pandas
numpy
tensorflow>=2.19
```

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/ARTEMISFOWL-01/bert_clickbait.git
cd bert_clickbait
```

2. Install required packages:
```bash
pip install transformers datasets torch scikit-learn pandas numpy
```

3. For GPU support:
```bash
pip install transformers[torch]
```

## 📚 Dataset

The project expects a CSV file with headlines and clickbait labels:
- **Column 1**: `headline` - The text of the headline
- **Column 2**: `clickbait` - Binary label (0 = not clickbait, 1 = clickbait)

The dataset is split into **80% training** and **20% validation** using a stratified split.

**Dataset Details:**
- Training samples: 25,600
- Validation samples: 6,400
- Total: 32,000 headlines

## 💡 How It Works

### 1. **Data Preparation**
   - Load headlines from CSV
   - Split into train/test sets (80/20)
   - Convert labels to integer format

### 2. **Tokenization**
   - Use BERT tokenizer to convert text to tokens
   - Apply padding to max length
   - Truncate longer sequences

### 3. **Model Architecture**
   ```
   Input (Headline Text)
        ↓
   BERT Tokenizer (max_length=512)
        ↓
   BERT-base-uncased (Pre-trained)
        ↓
   Classification Head (2 labels)
        ↓
   Softmax
        ↓
   Output (Clickbait: 0 or 1)
   ```

### 4. **Training**
   - **Optimizer**: AdamW
   - **Learning Rate**: 2e-5
   - **Batch Size**: 8 (per device)
   - **Epochs**: 10 (with early stopping)
   - **Loss Function**: CrossEntropyLoss
   - **Validation Strategy**: Evaluate at end of each epoch

### 5. **Early Stopping**
   - Stops training if validation loss doesn't improve for 2 consecutive epochs
   - Saves the best model automatically

## 📊 Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | bert-base-uncased |
| Batch Size | 8 |
| Learning Rate | 2e-5 |
| Epochs | 10 |
| Evaluation Strategy | Epoch-based |
| Loss Function | CrossEntropyLoss |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Early Stopping Patience | 2 epochs |

## 📈 Evaluation Metrics

The model is evaluated using:
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall)

## 🔧 Usage

Run the Jupyter notebook in Google Colab:

```python
# Mount Google Drive to save/load data
from google.colab import drive
drive.mount('/content/drive')

# The notebook will:
# 1. Load the clickbait dataset
# 2. Prepare and tokenize the data
# 3. Load BERT model
# 4. Fine-tune on the dataset
# 5. Evaluate performance
# 6. Save the trained model
```

## 🎓 Key Learning Points

1. **Transfer Learning**: Leveraging pre-trained BERT instead of training from scratch
2. **Fine-tuning**: Adapting a general model to a specific task
3. **Tokenization**: Converting text to machine-readable format
4. **Attention Mechanisms**: Understanding how BERT processes bidirectional context
5. **Early Stopping**: Preventing overfitting during training
6. **Evaluation Metrics**: Using multiple metrics for comprehensive model assessment

## 📁 Output Structure

After training, the project saves:
- **Model**: Trained BERT model weights
- **Tokenizer**: Saved tokenizer configuration
- **Results**: Training logs and evaluation metrics
- **Checkpoints**: Model checkpoints for each epoch

## ⚠️ Notes

- The model requires GPU for efficient training (recommended for Colab)
- Training time varies based on hardware (typically 4-6 hours on GPU)
- Early stopping helps prevent overfitting
- The model can be fine-tuned further on new data

## 🔍 Possible Improvements

- Experiment with other BERT variants (DistilBERT, RoBERTa)
- Implement data augmentation techniques
- Use ensemble methods combining multiple models
- Add more sophisticated preprocessing
- Test on different datasets
- Fine-tune hyperparameters (learning rate, batch size, etc.)

## 📝 License

This project is open source and available for educational purposes.

## 👤 Author

**ARTEMISFOWL-01**

---

**Happy Learning!** 🎉 Feel free to fork, modify, and improve this project!
