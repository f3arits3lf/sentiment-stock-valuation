import streamlit as st
import yfinance as yf
from textblob import TextBlob
import requests
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
import os
import time

# Function to get stock data from Yahoo Finance
@st.cache_data
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.history(period='5y')  # Extended historical data for 5 years
    except Exception as e:
        st.error(f"Error retrieving stock data for {ticker}: {e}")
        return pd.DataFrame()

# Function to get financial statements for calculating FCF
def get_financial_data(ticker, user_input=False, user_operating_cash_flow=None, user_capital_expenditure=None):
    try:
        stock = yf.Ticker(ticker)
        if user_input:
            operating_cash_flow = user_operating_cash_flow
            capital_expenditure = user_capital_expenditure
        else:
            if 'Total Cash From Operating Activities' not in stock.cashflow.index or 'Capital Expenditures' not in stock.cashflow.index:
                st.warning("Required financial data is missing for the selected ticker.")
                return None
            operating_cash_flow = stock.cashflow.loc['Total Cash From Operating Activities'][0]
            capital_expenditure = stock.cashflow.loc['Capital Expenditures'][0]

        free_cash_flow = operating_cash_flow - abs(capital_expenditure)
        return free_cash_flow
    except Exception as e:
        st.error(f"Error retrieving financial data for {ticker}: {e}")
        return None

# Function to get additional financial metrics
def get_additional_financial_metrics(ticker):
    try:
        stock = yf.Ticker(ticker)
        pe_ratio = stock.info.get('trailingPE', None)
        debt_to_equity = stock.info.get('debtToEquity', None)
        roe = stock.info.get('returnOnEquity', None)
        beta = stock.info.get('beta', None)
        eps = stock.info.get('trailingEps', None)
        wacc = stock.info.get('wacc', 0.0815)  # Use market-based discount rate if available, else default to 8.15%
        return {
            'PE Ratio': pe_ratio,
            'Debt to Equity': debt_to_equity,
            'ROE': roe,
            'Beta': beta,
            'EPS': eps,
            'WACC': wacc
        }
    except Exception as e:
        st.error(f"Error retrieving additional financial metrics for {ticker}: {e}")
        return {}

# Function to get news articles using News API
@st.cache_data
def get_news_articles(ticker):
    api_key = "40714f13bb7a4f4f92df63f537b78eb7"
    query = f"{ticker} AND (earnings OR growth OR revenue OR forecast OR stock analysis)"
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=popularity&apiKey={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            news_articles = [article['description'] for article in articles if article['description']]
            return news_articles[:5]  # Limit to 5 articles
        elif response.status_code == 429:
            st.warning("Rate limit exceeded for News API. Please try again later.")
            return []
        else:
            st.error(f"Error fetching news articles: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error retrieving news articles for {ticker}: {e}")
        return []

# Function to analyze sentiment from text
def analyze_sentiment(text):
    try:
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        if np.isnan(sentiment):
            return 0
        return sentiment
    except Exception as e:
        st.error(f"Error analyzing sentiment: {e}")
        return 0

# Function to calculate discounted cash flow (DCF) with growth rate
def discounted_cash_flow(free_cash_flow, discount_rate, sentiment_score, growth_rate=0.03, years=5):
    try:
        sentiment_weight = 0.5  # Limit sentiment adjustment to reduce overreaction
        adjusted_rate = discount_rate * (1 - sentiment_weight * sentiment_score)  # Adjust discount rate proportionally by sentiment
        adjusted_rate = max(0.01, adjusted_rate)  # Prevent adjusted_rate from becoming negative or zero
        total_value = sum(free_cash_flow * (1 + growth_rate)**i / (1 + adjusted_rate)**i for i in range(1, years + 1))
        return total_value
    except Exception as e:
        st.error(f"Error calculating DCF valuation: {e}")
        return None

# Function to perform valuation using P/E multiple
def pe_valuation(pe_ratio, earnings_per_share):
    try:
        if pe_ratio is not None and earnings_per_share is not None:
            return pe_ratio * earnings_per_share
        return None
    except Exception as e:
        st.error(f"Error calculating P/E valuation: {e}")
        return None

# LSTM Model for Price Prediction
def predict_future_prices_lstm(ticker, days=30):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5y")  # Use longer history for better training data
        hist['Close'] = hist['Close'].astype(float)

        # Preparing data for LSTM
        data = hist[['Close']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Creating training dataset
        seq_length = 60
        X, y = [], []
        for i in range(seq_length, len(scaled_data)):
            X.append(scaled_data[i-seq_length:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Building the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))  # Increased units to better capture patterns
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Training the model
        model.fit(X, y, batch_size=32, epochs=20, verbose=1)  # Increased epochs for better training

        # Predicting future prices
        predictions = []
        last_sequence = scaled_data[-seq_length:]
        for _ in range(days):
            X_pred = np.reshape(last_sequence, (1, seq_length, 1))
            predicted_price = model.predict(X_pred)
            predictions.append(predicted_price[0, 0])
            last_sequence = np.append(last_sequence[1:], predicted_price[0, 0])

        predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        future_dates = pd.date_range(start=hist.index.max(), periods=days + 1, freq='B')[1:]

        return pd.DataFrame({'Date': future_dates, 'Predicted Price': predicted_prices.flatten()})
    except Exception as e:
        st.error(f"Error predicting future prices for {ticker}: {e}")
        return pd.DataFrame()

# Function to plot predicted prices
def plot_predictions(predictions_df):
    try:
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Date', y='Predicted Price', data=predictions_df, marker='o')
        plt.title('Future Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Predicted Price')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error plotting predictions: {e}")

# Streamlit app
def main():
    st.sidebar.title("Stock Analysis Options")
    analysis_option = st.sidebar.selectbox("Choose Analysis Type", ["Valuation", "Price Prediction"])

    st.title("Sentiment-Driven Stock Analysis Tool")

    # Input: Stock ticker symbol
    ticker = st.text_input("Enter Stock Ticker Symbol", "AAPL")
    if ticker:
        if analysis_option == "Valuation":
            # Option to enter financial data manually
            user_input = st.checkbox("Enter Operating Cash Flow and Capital Expenditures manually?")
            if user_input:
                user_operating_cash_flow = st.number_input("Enter Operating Cash Flow (in dollars):", min_value=0)
                user_capital_expenditure = st.number_input("Enter Capital Expenditures (in dollars):", min_value=0)
            else:
                user_operating_cash_flow = None
                user_capital_expenditure = None

            # Allow user to enter custom discount rate and growth rate or use market-based rates
            financial_metrics = get_additional_financial_metrics(ticker)
            discount_rate = st.number_input("Enter Discount Rate (as a decimal, e.g., 0.1 for 10%) or use market-based rate (WACC)", min_value=0.0, value=financial_metrics.get('WACC', 0.0815))
            growth_rate = st.number_input("Enter Growth Rate (as a decimal, e.g., 0.03 for 3%) or use analyst estimates", min_value=0.0, value=0.05)

            # Step 1: Get Stock Data
            st.header("Stock Data")
            stock_data = get_stock_data(ticker)
            if not stock_data.empty:
                st.write(stock_data)

            # Step 2: Get Sentiment Data
            st.header("Sentiment Analysis")
            news_articles = get_news_articles(ticker)
            sentiment_scores = [analyze_sentiment(article) for article in news_articles]
            if len(sentiment_scores) > 0:
                average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                st.write(f"Average Sentiment Score: {average_sentiment:.2f}")
            else:
                average_sentiment = 0
                st.write("No news articles found for sentiment analysis.")

            # Step 3: Get Additional Financial Metrics
            st.header("Additional Financial Metrics")
            if financial_metrics:
                st.write(financial_metrics)

            # Allow user to enter missing financial metrics manually if unavailable
            if financial_metrics.get('PE Ratio') is None:
                financial_metrics['PE Ratio'] = st.number_input("Enter P/E Ratio (if unavailable):", min_value=0.0)
            if financial_metrics.get('EPS') is None:
                financial_metrics['EPS'] = st.number_input("Enter Earnings Per Share (EPS) (if unavailable):", min_value=0.0)

            # Step 4: Perform Valuation
            st.header("Stock Valuation")
            valuation_method = st.selectbox("Select Valuation Method", ["DCF", "P/E", "Both"])

            free_cash_flow = get_financial_data(ticker, user_input, user_operating_cash_flow, user_capital_expenditure)
            if valuation_method in ["DCF", "Both"]:
                if free_cash_flow is not None:
                    dcf_valuation = discounted_cash_flow(free_cash_flow, discount_rate, average_sentiment, growth_rate)
                    if dcf_valuation is not None:
                        st.write(f"DCF Valuation (Adjusted with Sentiment and Growth Rate): ${dcf_valuation:.2f}")
                else:
                    st.write("Unable to retrieve sufficient financial data for DCF valuation. Please enter the data manually if available.")

            if valuation_method in ["P/E", "Both"]:
                st.header("P/E Valuation")
                pe_valuation_value = pe_valuation(financial_metrics['PE Ratio'], financial_metrics['EPS'])
                if pe_valuation_value is not None:
                    st.write(f"P/E Valuation: ${pe_valuation_value:.2f}")
                else:
                    st.write("Unable to calculate P/E valuation due to missing P/E ratio or EPS data.")

        elif analysis_option == "Price Prediction":
            st.header("Future Price Prediction")
            days = st.number_input("Enter number of days to predict (e.g., 30):", min_value=1, value=30)
            prediction_method = st.selectbox("Select Prediction Method", ["LSTM"])

            if prediction_method == "LSTM":
                future_prices_df = predict_future_prices_lstm(ticker, days)
                if not future_prices_df.empty:
                    st.write(future_prices_df)
                    plot_predictions(future_prices_df)

if __name__ == "__main__":
    main()
