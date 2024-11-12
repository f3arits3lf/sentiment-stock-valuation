import streamlit as st
import yfinance as yf
from textblob import TextBlob
import requests
import pandas as pd
from googlesearch import search
from bs4 import BeautifulSoup

# Function to get stock data from Yahoo Finance
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    return stock.history(period='1y')  # Historical data for 1 year

# Function to get news articles using Google search
def get_news_articles(ticker):
    query = f"{ticker} stock news"
    news_links = [url for url in search(query, num=5, stop=5, pause=2)]
    news_articles = []

    for link in news_links:
        try:
            response = requests.get(link, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                paragraphs = soup.find_all('p')
                article_text = ' '.join([para.get_text() for para in paragraphs])
                news_articles.append(article_text[:1000])  # Limit text length for simplicity
        except Exception as e:
            print(f"Error retrieving article from {link}: {e}")

    return news_articles

# Function to analyze sentiment from text
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Function to calculate discounted cash flow (DCF)
def discounted_cash_flow(cash_flows, discount_rate, sentiment_score):
    adjusted_rate = discount_rate * (1 - sentiment_score)  # Example adjustment
    total_value = sum(cf / (1 + adjusted_rate)**i for i, cf in enumerate(cash_flows, start=1))
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
            st.write("No news articles found for sentiment analysis.")

        # Step 3: Perform Valuation
        st.header("Stock Valuation")
        cash_flows = [100, 150, 200, 250]  # Example cash flows
        discount_rate = 0.1  # Example discount rate
        valuation = discounted_cash_flow(cash_flows, discount_rate, average_sentiment if sentiment_scores else 0)
        st.write(f"Valuation (Adjusted with Sentiment): ${valuation:.2f}")

if __name__ == "__main__":
    main()
