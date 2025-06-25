import torch
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from modules.functions import *
from modules.modules import *
# ------------------------------------- Model Prediction Functions -------------------------------------

def predict_next_day_gold_price_arimax(df: pd.DataFrame, model) -> float:
    """
    Predict next day's gold price using ARIMAX with technical indicators.

    Returns:
        tuple: (next_day_price)
    """
    exog_cols = [
        'Returns', 'MA_5', 'MA_20', 'MA_50', 'Volatility',
        'RSI', 'BB_upper', 'BB_lower', 'BB_width',
        'BB_position', 'Sentiment',
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'Momentum_10', 'ROC_10'
    ]

    for col in exog_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df[['Close'] + exog_cols].dropna()
    df = df.asfreq('B')
    df.ffill(inplace=True)

    y = df['Close']
    exog = df[exog_cols]

    # -------------------------------
    # Forecast Next Price
    # -------------------------------
    next_exog = exog.iloc[[-1]].values
    predicted_price = model.forecast(steps=1, exog=next_exog).iloc[0]

    return predicted_price

def predict_next_day_gold_price_xgboost(gold: pd.DataFrame, model) -> float:
    """
    Predict next day's gold price using XGBoost with technical indicators.
    Returns:
        tuple: (next_day_price)
    """

    feature_cols = [
        'Returns', 'MA_5', 'MA_20', 'MA_50', 'Volatility',
        'RSI', 'BB_upper', 'BB_lower', 'BB_width',
        'BB_position', 'Sentiment',
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'Momentum_10', 'ROC_10'
    ]

    gold_clean = gold[['Close'] + feature_cols].copy().dropna()
    gold_clean['Close_pct_change_1'] = gold_clean['Close'].pct_change(1)
    gold_clean['Close_pct_change_2'] = gold_clean['Close'].pct_change(2)
    gold_clean['Close_pct_change_3'] = gold_clean['Close'].pct_change(3)
    gold_clean['Close_rolling_std_5'] = gold_clean['Close'].rolling(5).std()
    gold_clean['Close_rolling_std_10'] = gold_clean['Close'].rolling(10).std()
    gold_clean['Close_vs_MA5'] = (gold_clean['Close'] - gold_clean['MA_5']) / gold_clean['MA_5']
    gold_clean['Close_vs_MA20'] = (gold_clean['Close'] - gold_clean['MA_20']) / gold_clean['MA_20']
    feature_cols_extended = feature_cols + [
        'Close_pct_change_1', 'Close_pct_change_2', 'Close_pct_change_3',
        'Close_rolling_std_5', 'Close_rolling_std_10',
        'Close_vs_MA5', 'Close_vs_MA20']
    gold_clean = gold_clean.dropna()
    gold_clean['Target_pct_change'] = gold_clean['Close'].pct_change().shift(-1)
    gold_clean['Target_price'] = gold_clean['Close'].shift(-1)
    gold_clean = gold_clean.dropna()
    gold_clean = gold_clean[
        (np.isfinite(gold_clean['Target_pct_change'])) &
        (np.abs(gold_clean['Target_pct_change']) < 1.0)
    ]

    # Predict next day
    latest_features = gold_clean[feature_cols_extended].iloc[[-1]]
    latest_price = gold_clean['Close'].iloc[-1]
    scaler = RobustScaler().fit(gold_clean[feature_cols_extended])
    latest_scaled = scaler.transform(latest_features)
    next_day_pct_change = model.predict(latest_scaled)[0]
    next_day_price = latest_price * (1 + next_day_pct_change)

    return next_day_price

def predict_next_day_gold_price_rf(gold: pd.DataFrame, model) -> float:
    """
    Predict next day's gold price using Random Forest with enhanced features.
    Saves model daily and loads if already exists. Returns price, model, and percentage change.
    """

    # Feature Engineering
    feature_cols = [
        'Returns', 'MA_5', 'MA_20', 'MA_50', 'Volatility',
        'RSI', 'BB_upper', 'BB_lower', 'BB_width',
        'BB_position', 'Sentiment'
    ]

    gold_clean = gold[['Close'] + feature_cols].copy()
    gold_clean = gold_clean.dropna()
    gold_clean['Close_pct_change_1'] = gold_clean['Close'].pct_change(1)
    gold_clean['Close_pct_change_2'] = gold_clean['Close'].pct_change(2)
    gold_clean['Close_pct_change_3'] = gold_clean['Close'].pct_change(3)
    gold_clean['Close_rolling_std_5'] = gold_clean['Close'].rolling(5).std()
    gold_clean['Close_rolling_std_10'] = gold_clean['Close'].rolling(10).std()
    gold_clean['Close_vs_MA5'] = (gold_clean['Close'] - gold_clean['MA_5']) / gold_clean['MA_5']
    gold_clean['Close_vs_MA20'] = (gold_clean['Close'] - gold_clean['MA_20']) / gold_clean['MA_20']
    gold_clean['Price_momentum_3'] = gold_clean['Close'] / gold_clean['Close'].shift(3) - 1
    gold_clean['Price_momentum_5'] = gold_clean['Close'] / gold_clean['Close'].shift(5) - 1

    feature_cols_extended = feature_cols + [
        'Close_pct_change_1', 'Close_pct_change_2', 'Close_pct_change_3',
        'Close_rolling_std_5', 'Close_rolling_std_10',
        'Close_vs_MA5', 'Close_vs_MA20',
        'Price_momentum_3', 'Price_momentum_5']

    gold_clean.dropna(inplace=True)
    gold_clean['Target_pct_change'] = gold_clean['Close'].pct_change().shift(-1)
    gold_clean['Target_price'] = gold_clean['Close'].shift(-1)
    gold_clean.dropna(inplace=True)
    gold_clean = gold_clean[(np.abs(gold_clean['Target_pct_change']) < 1.0)]

    # Predict next day
    latest_features = gold_clean[feature_cols_extended].iloc[[-1]]
    latest_price = gold_clean['Close'].iloc[-1]
    scaler = RobustScaler().fit(gold_clean[feature_cols_extended])
    latest_scaled = scaler.transform(latest_features)
    next_day_pct_change = model.predict(latest_scaled)[0]
    next_day_price = latest_price * (1 + next_day_pct_change)

    return next_day_price


def predict_next_day_gold_price_lstm(gold: pd.DataFrame, device, model=None) -> float:
    
    sequence_length = 10 ## Based on the model's training sequence length
    
    # -------------------------------
    # Feature Setup
    # -------------------------------
    feature_cols = [
        'Returns', 'MA_5', 'MA_20', 'MA_50', 'Volatility',
        'RSI', 'BB_upper', 'BB_lower', 'BB_width',
        'BB_position', 'Sentiment'
    ]

    gold = gold[['Close'] + feature_cols].dropna()
    gold = gold.asfreq('B')
    gold.ffill(inplace=True)
    gold['Target'] = gold['Close'].shift(-1)
    gold.dropna(inplace=True)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(gold[feature_cols])
    y_scaled = scaler.fit_transform(gold[['Target']])

    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i + sequence_length])
        y_seq.append(y_scaled[i + sequence_length])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # -------------------------------
    # Forecast
    # -------------------------------
    model.eval()
    last_seq = torch.tensor(X_scaled[-sequence_length:], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        next_pred = model(last_seq).cpu().numpy()

    predicted_price = scaler.inverse_transform(
        np.concatenate([np.zeros((1, len(feature_cols))), next_pred], axis=1)
    )[:, -1][0]

    return predicted_price


def predict_next_day_gold_price_ensemble(
    ensemble_model: dict,
    arimax_pred: float,
    xgb_pred: float,
    rf_pred: float,
    lstm_pred: float,
    llm_pred: float
):
    """
    Use a pre-trained ensemble model (results hash) and new predictions to create a results hash.
    Only updates the 'individual_predictions' and recalculates all ensemble outputs.
    """
    # Extract weights and metadata from the loaded ensemble model
    weights_used = ensemble_model.get('weights_used', {})
    meta_weights = weights_used.get('meta_weights', {
        'simple': 0.1, 'weighted': 0.25, 'sentiment': 0.2, 'volatility': 0.25, 'trend': 0.2
    })
    norm_weights = weights_used.get('normalized_weights', {
        'arimax': 0.15, 'xgboost': 0.25, 'rf': 0.20, 'lstm': 0.20, 'llm': 0.20
    })
    vol_weights = weights_used.get('volatility_weights', {
        'arimax': 0.25, 'xgboost': 0.20, 'rf': 0.20, 'lstm': 0.15, 'llm': 0.20
    })
    trend_weights = weights_used.get('trend_weights', {
        'arimax': 0.15, 'xgboost': 0.25, 'rf': 0.20, 'lstm': 0.20, 'llm': 0.20
    })

    # Extract metadata for calculation
    metadata = ensemble_model.get('metadata', {})
    current_price = metadata.get('current_price', 0)
    current_sentiment = metadata.get('current_sentiment', 0)

    model_names = ['arimax', 'xgboost', 'rf', 'lstm', 'llm']
    model_preds = [arimax_pred, xgb_pred, rf_pred, lstm_pred, llm_pred]

    # 1. Simple average
    simple_avg = np.mean(model_preds)

    # 2. Weighted average (sentiment-boosted)
    weighted_avg = sum(norm_weights[k] * p for k, p in zip(model_names, model_preds))

    # 3. Sentiment-adjusted
    sentiment_factor = 1 + 0.02 * current_sentiment if abs(current_sentiment) > 0.1 else 1.0
    sentiment_adjusted = weighted_avg * sentiment_factor

    # 4. Volatility-weighted
    volatility_weighted = sum(vol_weights[k] * p for k, p in zip(model_names, model_preds))

    # 5. Trend-following
    trend_following = sum(trend_weights[k] * p for k, p in zip(model_names, model_preds))

    # 6. Meta-Ensemble
    meta_ensemble = (
        meta_weights['simple'] * simple_avg +
        meta_weights['weighted'] * weighted_avg +
        meta_weights['sentiment'] * sentiment_adjusted +
        meta_weights['volatility'] * volatility_weighted +
        meta_weights['trend'] * trend_following
    )

    pct_changes = {
        'simple_avg': (simple_avg - current_price) / current_price * 100 if current_price else 0,
        'weighted_avg': (weighted_avg - current_price) / current_price * 100 if current_price else 0,
        'sentiment_adjusted': (sentiment_adjusted - current_price) / current_price * 100 if current_price else 0,
        'volatility_weighted': (volatility_weighted - current_price) / current_price * 100 if current_price else 0,
        'trend_following': (trend_following - current_price) / current_price * 100 if current_price else 0,
        'meta_ensemble': (meta_ensemble - current_price) / current_price * 100 if current_price else 0
    }

    results = {
        'predictions': {
            'simple_average': simple_avg,
            'weighted_average': weighted_avg,
            'sentiment_adjusted': sentiment_adjusted,
            'volatility_weighted': volatility_weighted,
            'trend_following': trend_following,
            'meta_ensemble': meta_ensemble
        },
        'percentage_changes': pct_changes,
        'weights_used': weights_used,
        'metadata': metadata,
        'model_info': {
            **ensemble_model.get('model_info', {}),
            'created_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'individual_predictions': {
                'arimax': arimax_pred,
                'xgboost': xgb_pred,
                'random_forest': rf_pred,
                'lstm': lstm_pred,
                'llm': llm_pred
            }
        }
    }
    return results

def generate_news_input(device,news_data_csv,finbert_model,news_data_with_sentiment_csv):

    batch_predict_and_update_csv(news_data_csv,finbert_model,news_data_with_sentiment_csv)

    df_raw = pd.read_csv(news_data_with_sentiment_csv)
    df_processed = preprocess_dataset(df_raw)
    df_processed = generate_topic_encodings(df_processed)
    final_df = get_sentiment_combined_encodings(df_processed)
    
    #Convert embeddings to required dimension.
    # encodings = torch.tensor(np.random.rand(1, 9, 512).astype(np.float32), dtype=torch.float32).to(device)
    encodings = np.array(list(final_df.sentiment_combined_encodings), dtype=np.float32)
    encodings = torch.tensor(encodings.reshape(1,*encodings.shape), dtype=torch.float32).to(device)
    
    mask = torch.tensor(np.ones((encodings.shape[:2])).astype(np.float32), dtype=torch.float32).to(device)

    return encodings, mask

# 1. Define a function to load the SetTransformer model and weights
def load_news_llm_model( device, model_path):
    news_model = SetTransformer(
        dim_input=512,
        num_outputs=1,
        dim_output=1,
        num_inds=32,
        dim_hidden=128,
        num_heads=4,
        ln=True
    ).to(device)
    if os.path.exists(model_path):
        news_model, _ = load_checkpoint(model_path, news_model, device)
    return news_model