import requests
import pandas as pd
from datetime import datetime
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# --- Setup NLTK & VADER ---
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# --- Fetch Reddit posts using Pushshift ---
def fetch_reddit_posts(query, subreddit, start_date, end_date, max_results=1000):
    url = "https://api.pushshift.io/reddit/search/submission/"
    start_epoch = int(pd.to_datetime(start_date).timestamp())
    end_epoch = int(pd.to_datetime(end_date).timestamp())

    params = {
        'q': query,
        'subreddit': subreddit,
        'after': start_epoch,
        'before': end_epoch,
        'size': 100,
        'sort': 'desc'
    }

    all_data = []
    while len(all_data) < max_results:
        res = requests.get(url, params=params)
        data = res.json().get('data', [])
        if not data:
            break
        all_data.extend(data)
        params['before'] = data[-1]['created_utc']
    
    df = pd.DataFrame([{
        'date': datetime.utcfromtimestamp(d['created_utc']).date(),
        'title': d.get('title', ''),
        'selftext': d.get('selftext', ''),
        'url': d.get('full_link', ''),
        'score': d.get('score', 0)
    } for d in all_data])

    return df

# --- Preprocessing + Sentiment ---
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower()

def analyze_sentiment(text):
    score = sia.polarity_scores(text)
    compound = score['compound']
    if compound > 0.05:
        return 'positive'
    elif compound < -0.05:
        return 'negative'
    else:
        return 'neutral'

# --- Run ---
if __name__ == "__main__":
    df = fetch_reddit_posts(
        query='bitcoin',
        subreddit='Bitcoin',
        start_date='2025-01-01',
        end_date='2025-03-31',
        max_results=500
    )

    print(f"✅ Fetched {len(df)} posts. Analyzing sentiment...")

    df['text'] = (df['title'] + ' ' + df['selftext']).fillna('')
    df['clean_text'] = df['text'].apply(clean_text)
    df['sentiment'] = df['clean_text'].apply(analyze_sentiment)

    df.to_csv("reddit_bitcoin_sentiment_2025.csv", index=False)
    print("💾 Saved to 'reddit_bitcoin_sentiment_2025.csv'")
    print(df[['date', 'sentiment']].value_counts().sort_index())
