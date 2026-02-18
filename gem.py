import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

# 1. Configuration
FILE_PATH = 'watch-history.json'

def analyze_youtube_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    # Create DataFrame
    df = pd.DataFrame(data)

    # --- Data Cleaning ---
    # FIX: Use format='ISO8601' to handle varying timestamp formats
    # errors='coerce' will turn any unreadable dates into "NaT" (Not a Time)
    df['time'] = pd.to_datetime(df['time'], format='ISO8601', errors='coerce')
    
    # Drop rows where time couldn't be parsed
    df = df.dropna(subset=['time'])

    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour
    
    # Clean Title: Remove the "Watched " prefix
    # Use fillna('') to avoid errors if a title is missing
    df['clean_title'] = df['title'].fillna('').str.replace('Watched ', '', regex=False)

    # Extract Channel Name safely
    def get_channel(subtitles):
        if isinstance(subtitles, list) and len(subtitles) > 0:
            return subtitles[0].get('name', 'Unknown')
        return 'Unknown'
    df['channel'] = df['subtitles'].apply(get_channel)

    # --- Visualization ---

    # Plot 1: Daily Activity
    daily_counts = df.groupby('date').size()
    if not daily_counts.empty:
        plt.figure(figsize=(12, 6))
        daily_counts.plot(kind='line', color='#FF0000', linewidth=2)
        plt.title('Daily YouTube Activity', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Number of Videos')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('daily_activity.png')

    # Plot 2: Hourly Distribution
    hourly_counts = df.groupby('hour').size().reindex(range(24), fill_value=0)
    plt.figure(figsize=(10, 5))
    hourly_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Watch Activity by Hour of Day', fontsize=14)
    plt.xlabel('Hour (24-hour clock)')
    plt.ylabel('Total Videos Watched')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('hourly_distribution.png')

    # Plot 3: Keyword Analysis
    stop_words = set(['a', 'an', 'the', 'in', 'on', 'at', 'for', 'with', 'and', 'or', 'of', 'to', 'how', 'is', 'it', 'my', 'video', 'you', 'me', 'this', 'that', 'was', 'are'])
    all_titles = " ".join(df['clean_title'].astype(str).tolist()).lower()
    words = re.findall(r'\w+', all_titles)
    keywords = [w for w in words if w not in stop_words and len(w) > 3]
    
    top_keywords = Counter(keywords).most_common(15)
    if top_keywords:
        kw_df = pd.DataFrame(top_keywords, columns=['Keyword', 'Frequency']).sort_values(by='Frequency')
        plt.figure(figsize=(10, 6))
        plt.barh(kw_df['Keyword'], kw_df['Frequency'], color='forestgreen')
        plt.title('Top Keywords in Video Titles', fontsize=14)
        plt.xlabel('Frequency')
        plt.tight_layout()
        plt.savefig('top_keywords.png')

    # --- Text Output ---
    print("--- YOUTUBE WATCH HISTORY SUMMARY ---")
    print(f"Total Videos Analyzed: {len(df)}")
    if not df.empty:
        print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
        print("\nTop 10 Most Watched Channels:")
        print(df['channel'].value_counts().head(10))
        print("\nTop 10 Keywords Found in Titles:")
        for word, count in top_keywords[:10]:
            print(f"- {word}: {count} occurrences")
    else:
        print("No valid data found to analyze.")

# Run the analysis
analyze_youtube_data(FILE_PATH)
