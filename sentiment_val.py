import streamlit as st
import yfinance as yf
from textblob import TextBlob
import requests
import pandas as pd
from bs4 import BeautifulSoup

# Function to get stock data from Yahoo Finance
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    return stock.history(period='1y')  # Historical data for 1 year

# Function to get financial statements for calculating FCF
def get_financial_data(ticker):
    stock = yf.Ticker(ticker)
    try:
        if 'Total Cash From Operating Activities' in stock.cashflow.index:
            operating_cash_flow = stock.cashflow.loc['Total Cash From Operating Activities'][0]
        else:
            print("Operating cash flow data is missing. Using an estimated default value.")
            operating_cash_flow = 2000000000  # More conservative default value for operating cash flow

        if 'Capital Expenditures' in stock.cashflow.index:
            capital_expenditure = stock.cashflow.loc['Capital Expenditures'][0]
        else:
            print("Capital expenditures data is missing. Using an estimated default value.")
            capital_expenditure = 500000000  # More conservative default value for capital expenditures

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
        return {
            'PE Ratio': pe_ratio,
            'Debt to Equity': debt_to_equity,
            'ROE': roe,
            'Beta': beta
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

# Function to calculate discounted cash flow (DCF)
def discounted_cash_flow(free_cash_flow, discount_rate, sentiment_score, years=5):
    adjusted_rate = discount_rate * (1 - sentiment_score)  # Example adjustment
    adjusted_rate = max(0.01, adjusted_rate)  # Prevent adjusted_rate from becoming negative or zero
    total_value = sum(free_cash_flow / (1 + adjusted_rate)**i for i in range(1, years + 1))
    return total_value

# Streamlit app
def main():
    st.title("Sentiment-Driven Stock Valuation Tool")

    # Input: Stock ticker symbol
    ticker = st.text_input("Enter Stock Ticker Symbol", "AAPL")
    if ticker:
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
        financial_metrics = get_additional_financial_metrics(ticker)
        st.write(financial_metrics)

        # Step 4: Perform Valuation
        st.header("Stock Valuation")
        free_cash_flow = get_financial_data(ticker)
        if free_cash_flow is not None:
            discount_rate = 0.1  # Example discount rate
            valuation = discounted_cash_flow(free_cash_flow, discount_rate, average_sentiment)
            st.write(f"Valuation (Adjusted with Sentiment): ${valuation:.2f}")
        else:
            st.write("Unable to retrieve sufficient financial data for valuation.")

if __name__ == "__main__":
    main()
