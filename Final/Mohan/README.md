# Gold Price Prediction Using Sentiment Analysis (GoldBERT)

This module focuses on predicting **gold price movement** based on sentiment extracted from financial news using a fine-tuned **BERT model**, referred to as `GoldBERT`.

## 🚀 Overview

The model leverages the power of **transformer-based language models** to understand sentiment from financial news and uses this sentiment as a feature to predict the direction of gold prices. A pre-trained financial BERT model was fine-tuned on domain-specific data to classify sentiment related to gold.

## 🧠 Model Summary

- **Base Model**: `ProsusAI/finbert` (pretrained on financial text)
- **Fine-Tuned Model**: `GoldBERT`
  - Task: Sentiment Classification (Positive / Neutral / Negative)
  - Input: Financial news headlines or sentences
  - Output: Sentiment label, Probablities, Sentiment Confidence
    Example :
      - Text: gold futures edge up after two-session decline.
      - 
        Sentiment: positive  
        Confidence: 0.7530  
        Logits: [2.562419891357422, 1.4462203979492188, -5.02341365814209]  
        Class Probabilities: {'positive': '0.7530', 'neutral': '0.2466', 'negative': '0.0004'}
- **Gold Price Prediction**:
  - Sentiment scores (and optionally other features) are given as input to set transformer model to predict the direction of gold price.  
## 📊 Training Metrics

### Confusion Matrix – Sentiment Classification

> _(Placeholder: Insert confusion matrix image or values here)_

css
Copy
Edit
           Predicted
          P     N     Neut
Actual P x x x
N x x x
Neut x x x

markdown
Copy
Edit

## 🛠️ Technologies Used

- Python 3.11
- Transformers (`HuggingFace`)
- PyTorch
- Scikit-learn
- Pandas, NumPy
- Matplotlib / Seaborn
