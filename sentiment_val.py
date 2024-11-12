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

# Streamlit app
def main():
    st.title("Sentiment-Driven Stock Valuation Tool")

    # Input: Stock ticker symbol
    ticker = st.text_input("Enter Stock Ticker Symbol", "AAPL")
    if ticker:
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

if __name__ == "__main__":
    main()
