# %%
# This script installs all required libraries for data analysis, plotting, LLM workflows, and notebook imports.
# Note: The installation command is commented out to prevent accidental execution.
# --------------------------------------------------------------------------------

# Required Libraries:
# pandas: Data manipulation and analysis
# numpy: Numerical computations
# matplotlib: Data visualization
# yfinance: Downloading financial data from Yahoo Finance
# langchain: Building LLM-powered applications and chains
# import_ipynb: Importing Jupyter notebooks as Python modules
# scipy: Scientific computing (e.g., signal processing)
# statsmodels: Statistical modeling and time series analysis
# xgboost: Gradient boosting for machine learning
# selenium: Web scraping and browser automation
# webdriver_manager: Managing browser drivers for Selenium
# transformers: State-of-the-art NLP models
# peft: Parameter-efficient fine-tuning for transformers
# accelerate: Optimizing training and inference of models
# bitsandbytes: Efficient training of large models with 8-bit optimizers
# tensorflow: Deep learning framework
# torch: PyTorch deep learning framework
# tensorboard: Visualization tool for TensorFlow and PyTorch
# scikit-learn: Machine learning library for Python (version 1.6.1)

# Install all required libraries
# %pip install -U tensorflow pandas torch tensorboard numpy matplotlib yfinance langchain import_ipynb scipy statsmodels xgboost selenium webdriver_manager transformers peft accelerate bitsandbytes
# %pip install scikit-learn==1.6.1
# %pip install tensorflow-hub
# %pip install "numpy<2.0"
# %pip install --upgrade numpy
# %pip install gradio

# %%
import os
# -------------------------------------------------------------------------
#  LangChain Imports
# -------------------------------------------------------------------------
from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain, TransformChain
from langchain.schema.runnable import RunnableLambda, RunnableSequence
# -------------------------------------------------------------------------
# Other Imports
# -------------------------------------------------------------------------
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from datetime import datetime, timedelta
import statsmodels.api as sm
import torch
# -------------------------------------------------------------------------
#  Custom Imports ## It internally imports the modules & functions
# -------------------------------------------------------------------------
from modules.model_run_functions_old import *

# %%
#Get models if available, else train them.
def get_base_models (path,today,device):

    #If today's wights exists, then load them.
    if os.path.exists(f'{path}/LSTM/lstm_{today}.pt'):
        lstw_model           = LSTMModel(input_size=11).to(device)
        lstw_model.load_state_dict(torch.load(f'{path}/LSTM/lstm_{today}.pt', map_location=device))
        arimax_model         = sm.load_pickle(f'{path}/Arimax/arimax_{today}.pkl')
        random_forest_model  = sm.load_pickle(f'{path}/RandomForest/random_forest_{today}.pkl')
        xgboost_model        = sm.load_pickle(f'{path}/XGBoost/xgboost_{today}.pkl')
        ensemble_model       = sm.load_pickle(f'{path}/Final_Ensemble/ensemble_model_{today}.pkl')
    
    #If not then train models.
    else:

        today = "2025-06-23"
        ## TODO! : Replace with model training script!!!
        print(f"Could not load from {path}\n\n Loading from {today}")

        lstw_model           = LSTMModel(input_size=11).to(device)
        lstw_model.load_state_dict(torch.load(f'{path}/LSTM/lstm_{today}.pt', map_location=device))
        arimax_model         = sm.load_pickle(f'{path}/Arimax/arimax_{today}.pkl')
        random_forest_model  = sm.load_pickle(f'{path}/RandomForest/random_forest_{today}.pkl')
        xgboost_model        = sm.load_pickle(f'{path}/XGBoost/xgboost_{today}.pkl')
        ensemble_model       = sm.load_pickle(f'{path}/Final_Ensemble/ensemble_model_{today}.pkl')    

    news_model      = os.path.join(path, "/final_model.pth")

    return lstw_model,arimax_model,random_forest_model,xgboost_model,ensemble_model,news_model

# %%
def prepare_environment(inputs):
    WORKAREA = inputs["WORKAREA"]
    start = datetime(2010, 1, 1)
    end = datetime(2026, 1, 1)

    today = datetime.now().strftime("%Y-%m-%d")
    next_day = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Today's date: {today}")
    print(f"Next day's date: {next_day}")

    # Paths
    news_data_raw   = f'{WORKAREA}/Tarun/data/news_data_raw.csv'
    news_data_csv   = f"{WORKAREA}/Tarun/data/news_data_{today}.csv"
    news_data_with_sentiment_csv = f"{WORKAREA}/Tarun/data/news_data_with_sentiment_{today}.csv"
    gold_prices_csv = f"{WORKAREA}/Tarun/data/GOLDBEES_ETF_price_data_technical_indicators_sentiment.csv"
    
    if os.path.exists(gold_prices_csv):
        gold = pd.read_csv(gold_prices_csv, parse_dates=["Date"], index_col="Date")
    else:
        gold = generate_sentiment_from_trend_with_labels(
            add_technical_indicators(download_gold_prices(start, end))
        )

    current_price = gold["Close"].iloc[-1]
    print(f"Current Gold Price: {current_price}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    lstw_model, arimax_model, random_forest_model, xgboost_model, ensemble_model, news_model = get_base_models(
        f"{WORKAREA}/Tarun/Model", today, device
    )
    
    finbert_model = f"{WORKAREA}/Tarun/Model/finbert_best_model_merged"
    
    clean_and_prepare_articles(news_data_raw,news_data_csv)
    
    return {
        "start": start,
        "end": end,
        "today": today,
        "device": device,
        "next_day": next_day,
        "news_data_csv": news_data_csv,
        "news_data_with_sentiment_csv": news_data_with_sentiment_csv,
        "gold_prices_csv": gold_prices_csv,
        "df": gold,
        "current_price": current_price,        
        "lstw_model": lstw_model,
        "arimax_model": arimax_model,
        "random_forest_model": random_forest_model,
        "xgboost_model": xgboost_model,
        "news_model": news_model,
        "finbert_model" : finbert_model,
        "ensemble_model": ensemble_model
    }

# %%
## TODO [Yaswanth] : Replace this with today's news articles scraping.
# Extract news data
#news_df = extract_news_data(local_news=False)
#news_df.to_csv(news_data_raw, index=False)

# %%
# Orchestrate the workflow with SequentialChain
# --------------------------------------------------------------------
# Define prompt templates for each model prediction step
# Define TransformChains for each model prediction step using the existing functions
arimax_chain = TransformChain(
    input_variables=["df", "arimax_model"],
    output_variables=["predicted_price_arimax"],
    transform=lambda inputs: {
        "predicted_price_arimax": predict_next_day_gold_price_arimax(inputs["df"], inputs["arimax_model"])
    }
)

rf_chain = TransformChain(
    input_variables=["df", "random_forest_model"],
    output_variables=["predicted_price_rf"],
    transform=lambda inputs: {
        "predicted_price_rf": predict_next_day_gold_price_rf(inputs["df"], inputs["random_forest_model"])
    }
)

xgb_chain = TransformChain(
    input_variables=["df", "xgboost_model"],
    output_variables=["predicted_price_xgb"],
    transform=lambda inputs: {
        "predicted_price_xgb": predict_next_day_gold_price_xgboost(inputs["df"], inputs["xgboost_model"])
    }
)

lstm_chain = TransformChain(
    input_variables=["df", "device", "lstw_model"],
    output_variables=["predicted_price_lstw"],
    transform=lambda inputs: {
        "predicted_price_lstw": predict_next_day_gold_price_lstm(inputs["df"], inputs["device"], inputs["lstw_model"])
    }
)

def news_llm_transform(inputs):
    news_model = load_news_llm_model(inputs["device"], inputs["news_model"])
    encodings, mask = generate_news_input(
        inputs["device"],
        inputs["news_data_csv"],
        inputs["finbert_model"],
        inputs["news_data_with_sentiment_csv"]
    )
    with torch.no_grad():
        pred = news_model(encodings, mask=mask)
        if hasattr(pred, "item"):
            pred = pred.item()
    predicted_price = inputs["current_price"] * (1 + pred)
    return {"predicted_price_news_llm": predicted_price}

news_llm_chain = TransformChain(
    input_variables=["current_price", "device", "news_model", "finbert_model", "news_data_csv", "news_data_with_sentiment_csv"],
    output_variables=["predicted_price_news_llm"],
    transform=news_llm_transform
)

ensemble_chain = TransformChain(
    input_variables=[
        "ensemble_model",
        "predicted_price_arimax",
        "predicted_price_xgb",
        "predicted_price_rf",
        "predicted_price_lstw",
        "predicted_price_news_llm"
    ],
    output_variables=["ensemble_results"],
    transform=lambda inputs: {
        "ensemble_results": predict_next_day_gold_price_ensemble(
            inputs["ensemble_model"],
            inputs["predicted_price_arimax"],
            inputs["predicted_price_xgb"],
            inputs["predicted_price_rf"],
            inputs["predicted_price_lstw"],
            inputs["predicted_price_news_llm"]
        )
    }
)

# %%
# Compose the full sequence
full_seq_chain = SequentialChain(
    chains=[arimax_chain, rf_chain, xgb_chain, lstm_chain, news_llm_chain, ensemble_chain],
    input_variables=[
        "current_price", "device", "df", "next_day",
        "finbert_model", "news_data_csv", "news_data_with_sentiment_csv", "news_model", 
        "arimax_model", "random_forest_model", "xgboost_model", "lstw_model",
        "ensemble_model"        
    ],
    output_variables=[
        "predicted_price_arimax", "predicted_price_rf", "predicted_price_xgb", "predicted_price_lstw", "predicted_price_news_llm", 
        "ensemble_results"
    ]
)


# %%
def print_outputs(inputs):
    print ("---------------------------------------------------\n"
    f"Current Gold Price: {inputs['current_price']}\n"
    "---------------------------------------------------\n"
    f"Predictions for next day: {inputs['next_day']}\n"
    "---------------------------------------------------\n"
    f"ARIMAX: Predicted gold price: {inputs['predicted_price_arimax']}\n"
    f"Random Forest: Predicted gold price: {inputs['predicted_price_rf']}\n"
    f"XGBoost: Predicted gold price: {inputs['predicted_price_xgb']}\n"
    f"LSTM: Predicted gold price: {inputs['predicted_price_lstw']}\n"
    f"News LLM: Predicted gold price: {inputs['predicted_price_news_llm']}\n"
    "---------------------------------------------------\n"
    "Ensemble Model Results:\n"
    "---------------------------------------------------\n"
    f"Predicted Price: {inputs['ensemble_results']['predictions']['meta_ensemble']}\n"
    f"Percentage Change: {inputs['ensemble_results']['percentage_changes']['meta_ensemble']:.2f}%\n")

# %%
WORKAREA = "D:/CAREER/IISC_B/Academics/Courses/SEM_3/DA_225o/Project/DL_7_25/Final"

prepare_env = RunnableLambda(prepare_environment)
output_predictions =  RunnableLambda(print_outputs)

model_run = prepare_env | full_seq_chain | output_predictions

model_run.invoke({"WORKAREA": f"{WORKAREA}"})


