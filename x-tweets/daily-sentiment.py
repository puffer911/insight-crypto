import os
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import pandas as pd

# Load sentiment models
vader = SentimentIntensityAnalyzer()
roberta = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=-1)
distilbert = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)

files = [f"{i}.txt" for i in range(11, 31)]
data = []

def classify_tb(score):
    if score > 0.05: return 'positive'
    if score < -0.05: return 'negative'
    return 'neutral'

for filename in files:
    if not os.path.exists(filename):
        continue
    with open(filename, encoding="utf-8") as f:
        tweets = f.read().strip().split("\n\n")
        for tweet in tweets:
            parts = tweet.strip().split("\n", 1)
            if len(parts) != 2:
                continue
            meta, text = parts
            username, date = meta.split(" | ") if " | " in meta else ("unknown", "unknown")
            short_date = date[:10]

            vader_label = "neutral"
            try:
                compound = vader.polarity_scores(text)['compound']
                vader_label = "positive" if compound > 0.05 else "negative" if compound < -0.05 else "neutral"
            except:
                pass

            try:
                tb_label = classify_tb(TextBlob(text).sentiment.polarity)
            except:
                tb_label = "neutral"

            try:
                roberta_label = roberta(text[:512])[0]['label'].lower()
            except:
                roberta_label = "neutral"

            try:
                distilbert_label = distilbert(text[:512])[0]['label'].lower()
            except:
                distilbert_label = "neutral"

            data.append({
                "date": short_date,
                "vader": vader_label,
                "textblob": tb_label,
                "roberta": roberta_label,
                "distilbert": distilbert_label
            })

# Buat DataFrame
df = pd.DataFrame(data)

# --- DISABLED CSV OUTPUT ---
# df.to_csv("tweet_sentiment_results.csv", index=False)

# Rekap per hari dan per metode
for method in ["vader", "textblob", "roberta", "distilbert"]:
    print(f"\n📊 Daily Sentiment for {method.upper()}:")
    daily_summary = df.groupby("date")[method].value_counts().unstack().fillna(0).astype(int)
    print(daily_summary)
