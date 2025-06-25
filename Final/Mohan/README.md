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

> _<img width="240" alt="image" src="https://github.com/user-attachments/assets/68c3db29-aab0-4f2c-8fd0-029903362e12" /> <img width="240" alt="image" src="https://github.com/user-attachments/assets/2eb5d4c2-9dcf-433b-adc8-fab4e88903b8" />

🧪 Train Set Metrics  
| Metric       | Value      |
| ------------ | ---------- |
| **Classes**  | \[0, 1, 2] |
| **Accuracy** | 0.9526     |
| **F1 Score** | 0.9479     |

Classification Report (Train)  
| Label            | Precision | Recall | F1-Score   | Support  |
| ---------------- | --------- | ------ | ---------- | -------- |
| Positive         | 0.9646    | 0.9598 | 0.9622     | 3807     |
| Neutral          | 0.9187    | 0.9215 | 0.9201     | 2012     |
| Negative         | 0.9594    | 0.9632 | 0.9613     | 3311     |
| **Accuracy**     |           |        | **0.9526** | **9130** |
| **Macro Avg**    | 0.9476    | 0.9481 | 0.9479     | 9130     |
| **Weighted Avg** | 0.9526    | 0.9526 | 0.9526     | 9130     |

🧪 Validation Set Metrics
| Metric       | Value      |
| ------------ | ---------- |
| **Classes**  | \[0, 1, 2] |
| **Accuracy** | 0.9632     |
| **F1 Score** | 0.9594     |

Classification Report (Validation)  
| Label            | Precision | Recall | F1-Score   | Support  |
| ---------------- | --------- | ------ | ---------- | -------- |
| Positive         | 0.9665    | 0.9695 | 0.9680     | 952      |
| Neutral          | 0.9364    | 0.9364 | 0.9364     | 503      |
| Negative         | 0.9757    | 0.9722 | 0.9740     | 827      |
| **Accuracy**     |           |        | **0.9632** | **2282** |
| **Macro Avg**    | 0.9595    | 0.9594 | 0.9594     | 2282     |
| **Weighted Avg** | 0.9632    | 0.9632 | 0.9632     | 2282     |

📊 Cross-Fold Average Metrics  
| Metric                      | Value  |
| --------------------------- | ------ |
| Average Train Accuracy      | 0.9453 |
| Average Train F1 Score      | 0.9399 |
| Average Validation Accuracy | 0.9381 |
| Average Validation F1 Score | 0.9317 |


_

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
