import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

# Configuration
FILES = ['search-history.json', 'watch-history.json']
OUTPUT_TXT = 'ngram_results.txt'

def get_clean_tokens(titles, stop_words):
    token_lists = []
    for title in titles:
        # Standardize: remove prefixes, lowercase, and strip punctuation
        clean = str(title).replace('Searched for ', '').replace('Watched ', '')
        clean = re.sub(r'[^\w\s]', '', clean.lower())
        # Filter out common filler words
        words = [w for w in clean.split() if w not in stop_words and len(w) >= 3]
        if words:
            token_lists.append(words)
    return token_lists

def generate_ngrams(token_lists, n):
    ngrams = []
    for tokens in token_lists:
        if len(tokens) >= n:
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i:i+n])
                ngrams.append(ngram)
    return ngrams

def run_ngram_plotter():
    all_titles = []
    for f_path in FILES:
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_titles.extend([entry.get('title', '') for entry in data])
        except FileNotFoundError:
            print(f"Skipping {f_path}...")

    # Stopwords to filter out "noise"
    stop_words = set(['a', 'an', 'the', 'in', 'on', 'at', 'for', 'with', 'and', 'or', 'of', 'to', 'how', 'is', 'it', 'my', 'what', 'why', 'who', 'this', 'that', 'you', 'me', 'from', 'google', 'ads', 'youtube', 'video', 'playlist', 'can', 'be', 'as', 'your', 'will', 'if', 'about', 'just', 'shorts'])

    token_lists = get_clean_tokens(all_titles, stop_words)

    # Calculate Top 15 for Bigrams and Trigrams
    bigram_counts = Counter(generate_ngrams(token_lists, 2)).most_common(15)
    trigram_counts = Counter(generate_ngrams(token_lists, 3)).most_common(15)

    # 1. Write the full analysis to the text file
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        f.write("NGRAM FREQUENCY ANALYSIS\n" + "="*24 + "\n")
        f.write("\nTOP BIGRAMS (2-WORD PAIRS):\n")
        for p, c in bigram_counts: f.write(f"{p:<30} | {c}\n")
        f.write("\nTOP TRIGRAMS (3-WORD TRIPLETS):\n")
        for p, c in trigram_counts: f.write(f"{p:<30} | {c}\n")

    # 2. Plotting Bigrams
    if bigram_counts:
        bg_df = pd.DataFrame(bigram_counts, columns=['Phrase', 'Freq']).sort_values('Freq')
        plt.figure(figsize=(10, 6))
        plt.barh(bg_df['Phrase'], bg_df['Freq'], color='#34A853') # Google Green
        plt.title('Top 15 Two-Word Phrases (Bigrams)', fontsize=14)
        plt.xlabel('Frequency')
        plt.tight_layout()
        plt.savefig('bigram_plot.png')

    # 3. Plotting Trigrams
    if trigram_counts:
        tg_df = pd.DataFrame(trigram_counts, columns=['Phrase', 'Freq']).sort_values('Freq')
        plt.figure(figsize=(10, 6))
        plt.barh(tg_df['Phrase'], tg_df['Freq'], color='#EA4335') # Google Red
        plt.title('Top 15 Three-Word Phrases (Trigrams)', fontsize=14)
        plt.xlabel('Frequency')
        plt.tight_layout()
        plt.savefig('trigram_plot.png')

    print(f"Analysis complete. See {OUTPUT_TXT}, bigram_plot.png, and trigram_plot.png")

run_ngram_plotter()
