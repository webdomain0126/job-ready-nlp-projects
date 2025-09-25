import feedparser
import pandas as pd
from transformers import pipeline

# CNN RSS Feed
RSS_URL = "http://rss.cnn.com/rss/edition.rss"

def fetch_headlines():
    """Fetch top CNN headlines using RSS feed"""
    feed = feedparser.parse(RSS_URL)
    headlines = [entry.title for entry in feed.entries[:5]]
    return headlines

def summarize_headlines(headlines):
    """Summarize the fetched headlines"""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summaries = []
    for hl in headlines:
        summary = summarizer(hl, max_length=30, min_length=10, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    return summaries

def save_to_csv(headlines, summaries, filename="cnn_headlines.csv"):
    """Save headlines and summaries to a CSV file"""
    df = pd.DataFrame({
        "Headline": headlines,
        "Summary": summaries
    })
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"✅ Saved to {filename}")

if __name__ == "__main__":
    print("📰 Fetching top CNN headlines...")
    headlines = fetch_headlines()

    if headlines:
        print("\nTop Headlines:")
        for i, hl in enumerate(headlines, 1):
            print(f"{i}. {hl}")

        print("\n⚡ Summarizing...")
        summaries = summarize_headlines(headlines)

        # Save results
        save_to_csv(headlines, summaries)
    else:
        print("❌ No headlines found. Try again later.")




