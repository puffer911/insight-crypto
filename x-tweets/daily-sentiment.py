import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sns.set(style="whitegrid")

# Load sentiment models
vader = SentimentIntensityAnalyzer()
roberta = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=-1)
roberta_large = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english", device=-1)
bertweet = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis", device=-1)

files = [f"{i}.txt" for i in range(11, 31)]
data = []

def classify_tb(score):
    if score > 0.05: return 'positive'
    if score < -0.05: return 'negative'
    return 'neutral'

def map_roberta(label):
    return {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive"
    }.get(label, "neutral")

def map_bertweet(label):
    return {
        "pos": "positive",
        "neu": "neutral",
        "neg": "negative"
    }.get(label.lower(), "neutral")

# --- Sentiment Analysis with tqdm progress bar
for filename in files:
    if not os.path.exists(filename):
        continue
    with open(filename, encoding="utf-8") as f:
        tweets = f.read().strip().split("\n\n")
        print(f"\n📄 Processing {filename} ({len(tweets)} tweets)...")
        for tweet in tqdm(tweets, desc=filename, unit="tweet"):
            parts = tweet.strip().split("\n", 1)
            if len(parts) != 2:
                continue
            meta, text = parts
            username, date = meta.split(" | ") if " | " in meta else ("unknown", "unknown")
            short_date = date[:10]

            try:
                vader_score = vader.polarity_scores(text)['compound']
                vader_label = "positive" if vader_score > 0.05 else "negative" if vader_score < -0.05 else "neutral"
            except:
                vader_label = "neutral"

            try:
                tb_label = classify_tb(TextBlob(text).sentiment.polarity)
            except:
                tb_label = "neutral"

            try:
                roberta_label = map_roberta(roberta(text[:512])[0]['label'])
            except:
                roberta_label = "neutral"

            try:
                roberta_large_label = roberta_large(text[:512])[0]['label'].lower()
            except:
                roberta_large_label = "neutral"

            try:
                bertweet_label = map_bertweet(bertweet(text[:128])[0]['label'])  # limit to 128 tokens
            except:
                bertweet_label = "neutral"

            data.append({
                "date": short_date,
                "vader": vader_label,
                "textblob": tb_label,
                "roberta": roberta_label,
                "roberta_large": roberta_large_label,
                "bertweet": bertweet_label
            })

df = pd.DataFrame(data)

# --- Ambil tanggal dari tweet
target_dates = sorted(df['date'].unique())
start_unix = int(datetime.strptime(target_dates[0], "%Y-%m-%d").timestamp())
end_unix = int(datetime.strptime(target_dates[-1], "%Y-%m-%d").timestamp())

# --- Ambil harga BTC dari CoinGecko
print("\n🔄 Fetching Bitcoin price from CoinGecko...")
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
params = {
    "vs_currency": "usd",
    "from": start_unix,
    "to": end_unix + 86400,
}
res = requests.get(url, params=params)
prices = res.json()["prices"]

df_price = pd.DataFrame(prices, columns=["timestamp", "price"])
df_price["date"] = pd.to_datetime(df_price["timestamp"], unit="ms").dt.date
df_price = df_price.groupby("date").last().reset_index()
df_price["pct_change"] = df_price["price"].pct_change() * 100
df_price["log_return"] = np.log(df_price["price"] / df_price["price"].shift(1))
df_price = df_price[df_price["date"].isin(pd.to_datetime(target_dates).date)]
df_price.dropna(inplace=True)

# --- Ubah sentimen jadi skor numerik
sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
df_score = df.copy()
for col in ["vader", "textblob", "roberta", "roberta_large", "bertweet"]:
    df_score[col] = df_score[col].map(sentiment_map)

df_sentiment_daily = df_score.groupby("date").mean().reset_index()
df_sentiment_daily["date"] = pd.to_datetime(df_sentiment_daily["date"]).dt.date

# --- Gabung harga dan sentimen
df_merged = pd.merge(df_price, df_sentiment_daily, on="date", how="inner")

# --- Tampilkan tabel akhir
print("\n📅 Final Daily Table:")
print(df_merged[["date", "price", "pct_change", "log_return", "vader", "textblob", "roberta", "roberta_large", "bertweet"]])

# --- Korelasi Pearson
print("\n📈 Pearson Correlation (Sentiment vs Log Return):")
for method in ["vader", "textblob", "roberta", "roberta_large", "bertweet"]:
    corr, pval = pearsonr(df_merged["log_return"], df_merged[method])
    print(f"{method:<13}: r = {corr:.4f}, p = {pval:.4f}")

# --- LINE CHART
plt.figure(figsize=(12, 6))
plt.plot(df_merged["date"], df_merged["log_return"], label="Log Return", color="black", linewidth=2)
for method in ["vader", "textblob", "roberta", "roberta_large", "bertweet"]:
    plt.plot(df_merged["date"], df_merged[method], label=f"Sentiment: {method}")
plt.title("Daily Sentiment Scores vs Bitcoin Volatility")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig("linechart_sentiment_volatility.png")
plt.close()

# --- SCATTER PLOT + TRENDLINE
for method in ["vader", "textblob", "roberta", "roberta_large", "bertweet"]:
    plt.figure(figsize=(6, 4))
    sns.regplot(x=df_merged[method], y=df_merged["log_return"], scatter_kws={"s": 40}, line_kws={"color": "red"})
    plt.title(f"Sentiment vs Log Return: {method}")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Log Return")
    plt.tight_layout()
    plt.savefig(f"scatter_sentiment_{method}.png")
    plt.close()

# --- Save to CSV & Excel
df_merged.to_csv("sentiment_volatility.csv", index=False)
df_merged.to_excel("sentiment_volatility.xlsx", index=False)
print("\n✅ Data saved as CSV & Excel. Charts exported as PNG.")
