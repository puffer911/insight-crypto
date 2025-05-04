import tweepy
import pandas as pd
import re
import nltk
import time
import os
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datetime import datetime

# --- Load ENV ---
load_dotenv()
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
if not BEARER_TOKEN:
    raise ValueError("❌ TWITTER_BEARER_TOKEN tidak ditemukan di .env!")

# --- NLTK Setup ---
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --- Setup Tweepy Client ---
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# --- Ambil tweet dengan retry jika rate limit ---
def fetch_tweets(query, max_results=100):
    while True:
        try:
            tweets = client.search_recent_tweets(
                query=query,
                max_results=max_results,
                tweet_fields=['created_at', 'text', 'lang']
            )
            if not tweets.data:
                print("⚠️ Tidak ada tweet ditemukan.")
                return pd.DataFrame()

            data = []
            for tweet in tweets.data:
                if tweet.lang == 'en':
                    data.append({
                        'date': tweet.created_at.date(),
                        'text': tweet.text
                    })
            return pd.DataFrame(data)

        except tweepy.errors.TooManyRequests:
            print("⏳ Rate limit. Menunggu 15 menit...")
            time.sleep(900)

        except Exception as e:
            print(f"❌ Error: {e}")
            return pd.DataFrame()

# --- Preprocessing ---
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

# --- Eksekusi utama ---
if __name__ == "__main__":
    print("🔍 Mengambil tweet tentang Bitcoin...")
    df = fetch_tweets("bitcoin OR BTC -is:retweet", max_results=100)

    if df.empty:
        print("⚠️ Tidak ada data untuk diproses.")
    else:
        df['clean_text'] = df['text'].apply(preprocess)
        df.to_csv("bitcoin_tweets_clean.csv", index=False)
        print("✅ Disimpan ke 'bitcoin_tweets_clean.csv'")
        print(df[['date', 'clean_text']].head())
