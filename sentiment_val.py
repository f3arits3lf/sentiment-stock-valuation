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
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Function to get stock data from Yahoo Finance
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    return stock.history(period='1y')  # Historical data for 1 year

# Function to get financial statements for calculating FCF
def get_financial_data(ticker, user_input=False, user_operating_cash_flow=None, user_capital_expenditure=None):
    stock = yf.Ticker(ticker)
    try:
        if user_input:
            operating_cash_flow = user_operating_cash_flow
            capital_expenditure = user_capital_expenditure
        else:
            if 'Total Cash From Operating Activities' in stock.cashflow.index and 'Capital Expenditures' in stock.cashflow.index:
                operating_cash_flow = stock.cashflow.loc['Total Cash From Operating Activities'][0]
                capital_expenditure = stock.cashflow.loc['Capital Expenditures'][0]
            else:
                return None  # Return None if required financial data is missing

        free_cash_flow = operating_cash_flow - abs(capital_expenditure)
        return free_cash_flow
    except Exception as e:
        print(f"Error retrieving financial data: {e}")
        return None

# Function to get additional financial metrics
def get_additional_financial_metrics(ticker):
    stock = yf.Ticker(ticker)
    try:
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
        print(f"Error retrieving additional financial metrics: {e}")
        return {}

# Function to get news articles using News API
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
        else:
            print(f"Error fetching news articles: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error retrieving news articles: {e}")
        return []

# Function to analyze sentiment from text
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Function to calculate discounted cash flow (DCF) with growth rate
def discounted_cash_flow(free_cash_flow, discount_rate, sentiment_score, growth_rate=0.03, years=5):
    sentiment_weight = 0.5  # Limit sentiment adjustment to reduce overreaction
    adjusted_rate = discount_rate * (1 - sentiment_weight * sentiment_score)  # Adjust discount rate proportionally by sentiment
    adjusted_rate = max(0.01, adjusted_rate)  # Prevent adjusted_rate from becoming negative or zero
    total_value = sum(free_cash_flow * (1 + growth_rate)**i / (1 + adjusted_rate)**i for i in range(1, years + 1))
    return total_value

# Function to perform valuation using P/E multiple
def pe_valuation(pe_ratio, earnings_per_share):
    if pe_ratio is not None and earnings_per_share is not None:
        return pe_ratio * earnings_per_share
    return None

# Transformer Model for Price Prediction
class StockPriceDataset(Dataset):
    def __init__(self, data, seq_length=10):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.float32),
            torch.tensor(self.data[idx + self.seq_length, 0], dtype=torch.float32)  # Predicting only the 'Close' price
        )

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, nhead, hidden_dim, num_layers):
        super(SimpleTransformer, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(input_dim * seq_length, 1)  # Fully connected layer for output

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.view(x.size(1), -1)  # Flatten for fully connected layer
        return self.fc(x)

# Function to predict future prices using Transformer model
def predict_future_prices_transformer(ticker, days=30):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    hist['Close'] = hist['Close'].astype(float)

    # Adding additional features: Volume and Moving Average
    hist['Volume'] = hist['Volume'].astype(float)
    hist['MA10'] = hist['Close'].rolling(window=10).mean()
    hist.dropna(inplace=True)

    features = hist[['Close', 'Volume', 'MA10']].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Prepare Dataset
    seq_length = 10
    dataset = StockPriceDataset(features_scaled, seq_length)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Transformer model
    input_dim = features.shape[1]
    nhead = 2
    hidden_dim = 128
    num_layers = 2
    model = SimpleTransformer(input_dim, nhead, hidden_dim, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training the Transformer model
    epochs = 10
    model.train()
    for epoch in range(epochs):
        for X, y in dataloader:
            optimizer.zero_grad()
            X = X.squeeze(0).permute(1, 0)  # Transformer expects input in (seq_length, batch_size, features)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

    # Predicting future prices
    model.eval()
    with torch.no_grad():
        X_input = torch.tensor(features_scaled[-seq_length:], dtype=torch.float32).permute(1, 0).unsqueeze(1)
        predicted_prices = []
        for _ in range(days):
            y_pred = model(X_input)
            predicted_prices.append(y_pred.item())
            X_input = torch.cat((X_input[:, 1:, :], y_pred.unsqueeze(0).unsqueeze(0)), dim=1)

    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, input_dim))[:, 0]
    future_dates = [hist.index.max() + datetime.timedelta(days=i) for i in range(1, days + 1)]

    return pd.DataFrame({'Date': future_dates, 'Predicted Price': predicted_prices})

# Function to plot predicted prices
def plot_predictions(predictions_df):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Date', y='Predicted Price', data=predictions_df, marker='o')
    plt.title('Future Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Predicted Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

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
            st.write(stock_data)

            # Step 2: Get Sentiment Data
            st.header("Sentiment Analysis")
            news_articles = get_news_articles(ticker)
            sentiment_scores = [analyze_sentiment(article) for article in news_articles]
            if sentiment_scores:
                average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                st.write(f"Average Sentiment Score: {average_sentiment:.2f}")
            else:
                average_sentiment = 0
                st.write("No news articles found for sentiment analysis.")

            # Step 3: Get Additional Financial Metrics
            st.header("Additional Financial Metrics")
            st.write(financial_metrics)

            # Allow user to enter missing financial metrics manually if unavailable
            if financial_metrics['PE Ratio'] is None:
                financial_metrics['PE Ratio'] = st.number_input("Enter P/E Ratio (if unavailable):", min_value=0.0)
            if financial_metrics['EPS'] is None:
                financial_metrics['EPS'] = st.number_input("Enter Earnings Per Share (EPS) (if unavailable):", min_value=0.0)

            # Step 4: Perform Valuation
            st.header("Stock Valuation")
            valuation_method = st.selectbox("Select Valuation Method", ["DCF", "P/E", "Both"])

            free_cash_flow = get_financial_data(ticker, user_input, user_operating_cash_flow, user_capital_expenditure)
            if valuation_method in ["DCF", "Both"]:
                if free_cash_flow is not None:
                    dcf_valuation = discounted_cash_flow(free_cash_flow, discount_rate, average_sentiment, growth_rate)
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
            prediction_method = st.selectbox("Select Prediction Method", ["Transformer"])

            if prediction_method == "Transformer":
                future_prices_df = predict_future_prices_transformer(ticker, days)

            st.write(future_prices_df)
            plot_predictions(future_prices_df)

if __name__ == "__main__":
    main()
