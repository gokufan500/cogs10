import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

# Configuration
FILE_PATH = 'search-history.json'
OUTPUT_TXT = 'keyword_counts.txt'
OUTPUT_IMG = 'keyword_graph.png'

def analyze_and_graph_keywords(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error: {e}")
        return

    df = pd.DataFrame(data)

    # Clean the titles to isolate queries/topics
    df['clean_text'] = df['title'].str.replace('Searched for ', '', regex=False)
    df['clean_text'] = df['clean_text'].str.replace('Watched ', '', regex=False)

    # Process words
    all_text = " ".join(df['clean_text'].astype(str).tolist()).lower()
    words = re.findall(r'\w+', all_text)

    # Expanded filter list for cleaner results
    stop_words = set([
        'a', 'an', 'the', 'in', 'on', 'at', 'for', 'with', 'and', 'or', 'of', 'to', 
        'how', 'is', 'it', 'my', 'what', 'why', 'who', 'this', 'that', 'was', 'are',
        'you', 'me', 'from', 'google', 'ads', 'youtube', 'video', 'playlist', 'can', 
        'https', 'com'
    ])

    keywords = [w for w in words if w not in stop_words and len(w) >= 3]
    counts = Counter(keywords).most_common()

    # --- 1. Export to Text File ---
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        f.write("KEYWORD FREQUENCY ANALYSIS\n" + "="*26 + "\n")
        f.write(f"{'KEYWORD':<20} | {'FREQUENCY':<10}\n")
        f.write("-" * 35 + "\n")
        for word, freq in counts:
            f.write(f"{word:<20} | {freq:<10}\n")

    # --- 2. Generate Graph (Top 20 Keywords) ---
    top_20 = counts[:20]
    if top_20:
        # Convert to DataFrame for plotting
        kw_df = pd.DataFrame(top_20, columns=['Keyword', 'Frequency'])
        # Sort so the highest is at the top of the horizontal bar chart
        kw_df = kw_df.sort_values(by='Frequency', ascending=True)

        plt.figure(figsize=(12, 8))
        # Use Google's brand blue for the chart
        bars = plt.barh(kw_df['Keyword'], kw_df['Frequency'], color='#4285F4', edgecolor='black', alpha=0.8)
        
        # Add frequency labels to the end of each bar
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.3, bar.get_y() + bar.get_height()/2, 
                     f'{int(width)}', va='center', fontsize=10)

        plt.title('Top 20 Keywords in Personal Search/Watch History', fontsize=16, pad=20)
        plt.xlabel('Number of Occurrences', fontsize=12)
        plt.ylabel('Keyword', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_IMG)
        print(f"Graph successfully saved to: {OUTPUT_IMG}")
    
    print(f"Text summary successfully saved to: {OUTPUT_TXT}")

# Run the full analysis
analyze_and_graph_keywords(FILE_PATH)
