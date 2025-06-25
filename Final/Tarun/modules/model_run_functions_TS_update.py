#Import necessary packages.
#import tensorflow_hub as hub
import os
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from modules.modules import LSTMModel, GoldPriceDataset, SetTransformer
from modules.functions import *
# -------------------------------------------------------------------------

def predict_next_day_gold_price_arimax(df: pd.DataFrame, model_dir, arima_order=(1, 1, 1)):
    """
    Predict next day's gold price using ARIMAX with technical indicators.
    Loads today's model if exists, otherwise retrains and saves a new model.

    Returns:
        tuple: (next_day_price, model_fit, predicted_pct_change)
    """
    # -------------------------------
    # Step 1: Setup
    # -------------------------------
    os.makedirs(model_dir, exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    model_path = os.path.join(model_dir, f"arimax_{today_str}.pkl")

    # -------------------------------
    # Step 2: Define Exogenous Features
    # -------------------------------
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
    # Step 3: Check if today's model exists
    # -------------------------------
    if os.path.exists(model_path):
        print(f"Loading existing ARIMAX model for today: {model_path}")
        with open(model_path, "rb") as f:
            model_fit = pickle.load(f)
    else:
        print("No model found for today. Retraining ARIMAX model...")

        # Clean old models
        for fname in os.listdir(model_dir):
            if fname.startswith("arimax_") and fname.endswith(".pkl"):
                os.remove(os.path.join(model_dir, fname))

        # Train new model
        model = SARIMAX(endog=y, exog=exog, order=arima_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False, method='powell')

        with open(model_path, "wb") as f:
            pickle.dump(model_fit, f)
        print(f"Saved new ARIMAX model to: {model_path}")

    # -------------------------------
    # Forecast Next Price
    # -------------------------------
    next_exog = exog.iloc[[-1]].values
    predicted_price = model_fit.forecast(steps=1, exog=next_exog).iloc[0]

    return predicted_price


def predict_next_day_gold_price_xgboost(gold: pd.DataFrame, model_dir, test_size: float = 0.2, random_state: int = 42):
    
    os.makedirs(model_dir, exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    model_path = os.path.join(model_dir, f"xgboost_{today_str}.pkl")

    # Remove old models
    for fname in os.listdir(model_dir):
        if fname.startswith("xgboost_") and fname.endswith(".pkl") and fname != f"xgboost_{today_str}.pkl":
            os.remove(os.path.join(model_dir, fname))

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

    if os.path.exists(model_path):
        print(f"Loading XGBoost model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        X = gold_clean[feature_cols_extended]
        y_pct = gold_clean['Target_pct_change']

        split_idx = int(len(X) * (1 - test_size))
        X_train, y_pct_train = X.iloc[:split_idx], y_pct.iloc[:split_idx]

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=10,
            learning_rate=0.008,
            min_child_weight=1,
            subsample=0.95,
            colsample_bytree=0.9,
            reg_alpha=0.001,
            reg_lambda=0.01,
            gamma=0,
            random_state=random_state,
            objective='reg:squarederror',
            tree_method='hist')

        print("Training new XGBoost model...")
        model.fit(X_train_scaled, y_pct_train)

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved model to {model_path}")

    # Predict next day
    latest_features = gold_clean[feature_cols_extended].iloc[[-1]]
    latest_price = gold_clean['Close'].iloc[-1]
    scaler = RobustScaler().fit(gold_clean[feature_cols_extended])
    latest_scaled = scaler.transform(latest_features)
    next_day_pct_change = model.predict(latest_scaled)[0]
    next_day_price = latest_price * (1 + next_day_pct_change)

    return next_day_price


def predict_next_day_gold_price_rf(gold: pd.DataFrame, model_dir) -> float:
    """
    Predict next day's gold price using Random Forest with enhanced features.
    Saves model daily and loads if already exists. Returns price, model, and percentage change.
    """

    os.makedirs(model_dir, exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    model_path = os.path.join(model_dir, f"random_forest_{today_str}.pkl")

    # Clean old models
    for fname in os.listdir(model_dir):
        if fname.startswith("random_forest_") and fname.endswith(".pkl") and fname != f"random_forest_{today_str}.pkl":
            os.remove(os.path.join(model_dir, fname))

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

    if os.path.exists(model_path):
        print(f"Loading existing Random Forest model for today: {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        print("No model found for today. Training Random Forest model...")
        X = gold_clean[feature_cols_extended]
        y_pct = gold_clean['Target_pct_change']
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2,
            max_features='sqrt', bootstrap=True, random_state=random_state, n_jobs=-1)
        model.fit(X_scaled, y_pct)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved model to {model_path}")

    # Predict next day
    latest_features = gold_clean[feature_cols_extended].iloc[[-1]]
    latest_price = gold_clean['Close'].iloc[-1]
    scaler = RobustScaler().fit(gold_clean[feature_cols_extended])
    latest_scaled = scaler.transform(latest_features)
    next_day_pct_change = model.predict(latest_scaled)[0]
    next_day_price = latest_price * (1 + next_day_pct_change)

    return next_day_price

def predict_next_day_gold_price_lstm(gold: pd.DataFrame, model_dir, sequence_length=10, epochs=50, batch_size=16, lr=0.001):
 
    os.makedirs(model_dir, exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    model_path = os.path.join(model_dir, f"lstm_{today_str}.pt")
    
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
    # Step 2: Load or Train Model
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size=X_seq.shape[2]).to(device)

    if os.path.exists(model_path):
        print(f"Loading LSTM model from {model_path}")
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        print("No model found for today. Retraining LSTM model...")

        # Delete old models
        for f in os.listdir(model_dir):
            if f.startswith("lstm_") and f.endswith(".pt"):
                os.remove(os.path.join(model_dir, f))

        train_ds = GoldPriceDataset(X_seq, y_seq)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model.train()
        for epoch in range(epochs):
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                output = model(xb).squeeze()
                loss = criterion(output, yb.squeeze())
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch} - Loss: {loss.item():.6f}")

        torch.save(model.state_dict(), model_path)
        print(f"Saved new LSTM model to: {model_path}")

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


def generate_news_input(device,news_data_csv,gold_data_plain_csv,finbert_model,news_data_with_sentiment_csv):

    batch_predict_and_update_csv(news_data_csv,finbert_model,news_data_with_sentiment_csv)

    df_raw = pd.read_csv(news_data_with_sentiment_csv)
    df_gold = pd.read_csv(gold_data_plain_csv)
    df_processed = preprocess_dataset(df_raw)
    df_processed = generate_topic_encodings(df_processed)
    final_df = add_gold_price_change_with_weekend_handling(df_processed,df_gold)
    
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