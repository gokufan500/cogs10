#!/usr/bin/env python3
"""
YouTube Search History Sentiment Analyzer
Analyzes the emotional tone of your YouTube searches over time
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
from collections import Counter
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except:
    nltk.download('vader_lexicon')
    nltk.download('punkt')

def extract_search_history(html_file_path):
    """
    Extract search history from YouTube Takeout HTML file
    Returns: List of dictionaries with search info
    """
    print(f"Reading HTML file: {html_file_path}")
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print(f"HTML file size: {len(html_content)} characters")
    
    search_history = []
    
    # Look for search entries - YouTube Takeout typically has this structure
    # Find all entries that look like search queries with dates
    
    # Method 1: Look for specific search patterns
    print("Looking for search patterns...")
    
    # Common patterns in YouTube Takeout
    patterns = [
        # Pattern for search entries
        r'>([^<]*?Searched for[^<]*?)<.*?(\w{3}\s+\d{1,2},\s+\d{4},\s+\d{1,2}:\d{2}:\d{2}\s*[AP]M)',
        # Pattern with "Searched for" text
        r'Searched for[^>]*>([^<]+)<.*?(\w{3}\s+\d{1,2},\s+\d{4},\s+\d{1,2}:\d{2}:\d{2}\s*[AP]M)',
        # More generic pattern
        r'<div[^>]*>([^<]+?)</div>.*?(\w{3}\s+\d{1,2},\s+\d{4},\s+\d{1,2}:\d{2}:\d{2}\s*[AP]M)',
    ]
    
    for i, pattern in enumerate(patterns):
        entries = re.findall(pattern, html_content, re.IGNORECASE | re.DOTALL)
        print(f"Pattern {i+1} found {len(entries)} potential matches")
        
        for text, date_str in entries:
            try:
                # Clean the text
                clean_text = re.sub(r'<[^>]+>', '', text).strip()
                clean_text = re.sub(r'Searched for', '', clean_text, flags=re.IGNORECASE).strip()
                
                if not clean_text or len(clean_text) < 2:
                    continue
                
                # Parse date
                date_formats = [
                    '%b %d, %Y, %I:%M:%S %p %Z',
                    '%b %d, %Y, %I:%M:%S %p GMT',
                    '%b %d, %Y, %I:%M:%S %p',
                    '%b %d, %Y, %I:%M:%S'
                ]
                
                timestamp = None
                for fmt in date_formats:
                    try:
                        timestamp = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                
                if timestamp:
                    search_history.append({
                        'timestamp': timestamp,
                        'search_query': clean_text,
                        'date_str': date_str
                    })
                    
            except Exception as e:
                continue
    
    # Method 2: If no searches found, look for any entries with dates
    if not search_history:
        print("Trying alternative extraction method...")
        # Split by lines and look for date patterns
        lines = html_content.split('\n')
        date_pattern = r'(\w{3}\s+\d{1,2},\s+\d{4},\s+\d{1,2}:\d{2}:\d{2}\s*[AP]M)'
        
        for line in lines:
            date_match = re.search(date_pattern, line)
            if date_match:
                date_str = date_match.group(1)
                # Extract potential search text (text before the date)
                parts = line.split(date_str)
                if len(parts) > 1:
                    # Take text before the date
                    potential_text = parts[0].strip()
                    # Clean HTML tags
                    clean_text = re.sub(r'<[^>]+>', ' ', potential_text)
                    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                    
                    # Skip if too short or looks like navigation
                    if (len(clean_text) > 3 and 
                        not clean_text.startswith('http') and
                        'google' not in clean_text.lower() and
                        'youtube' not in clean_text.lower()):
                        
                        # Parse date
                        for fmt in ['%b %d, %Y, %I:%M:%S %p']:
                            try:
                                timestamp = datetime.strptime(date_str, fmt)
                                break
                            except ValueError:
                                continue
                        else:
                            continue
                        
                        search_history.append({
                            'timestamp': timestamp,
                            'search_query': clean_text[:200],
                            'date_str': date_str
                        })
    
    print(f"Total search entries found: {len(search_history)}")
    return search_history

def clean_search_query(query):
    """Clean a single search query"""
    if not isinstance(query, str):
        return ""
    
    # Remove URLs
    query = re.sub(r'https?://\S+', '', query)
    # Remove special characters (keep spaces and basic punctuation)
    query = re.sub(r'[^\w\s.,!?-]', ' ', query)
    # Remove extra spaces
    query = re.sub(r'\s+', ' ', query).strip()
    # Convert to lowercase
    query = query.lower()
    
    return query

def clean_search_dataframe(df):
    """Clean search queries in DataFrame and filter invalid ones"""
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Clean the queries
    df_clean['cleaned_query'] = df_clean['search_query'].apply(clean_search_query)
    
    # Filter out queries that are too short after cleaning
    df_clean = df_clean[df_clean['cleaned_query'].str.len() > 2]
    
    # Remove exact duplicates (same query and same timestamp)
    df_clean = df_clean.drop_duplicates(subset=['cleaned_query', 'timestamp'])
    
    print(f"After cleaning: {len(df_clean)} queries (removed {len(df) - len(df_clean)})")
    return df_clean

def analyze_sentiment_vader(queries):
    """Analyze sentiment using VADER (better for short text like search queries)"""
    print("Running VADER sentiment analysis...")
    sia = SentimentIntensityAnalyzer()
    sentiments = []
    
    for i, query in enumerate(queries):
        scores = sia.polarity_scores(query)
        
        # Determine sentiment category
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        sentiments.append({
            'query': query,
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'sentiment': sentiment
        })
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1} queries...")
    
    return pd.DataFrame(sentiments)

def categorize_search_topics(queries):
    """Categorize searches into broad topics"""
    print("Categorizing search topics...")
    
    # Expanded categories with more keywords
    categories = {
        'entertainment': ['movie', 'film', 'trailer', 'music', 'song', 'album', 'artist', 
                         'game', 'gaming', 'play', 'stream', 'episode', 'series', 'tv',
                         'netflix', 'spotify', 'trailer', 'music video', 'concert'],
        'educational': ['how to', 'tutorial', 'explain', 'learn', 'course', 'lesson', 
                       'education', 'study', 'research', 'science', 'history', 'math',
                       'physics', 'chemistry', 'biology', 'documentary', 'lecture'],
        'news': ['news', 'update', 'breaking', 'latest', 'current', 'politics', 'world',
                'election', 'trump', 'biden', 'president', 'government', 'bbc', 'cnn'],
        'shopping': ['buy', 'price', 'shop', 'store', 'deal', 'discount', 'review', 
                    'amazon', 'ebay', 'product', 'cheap', 'sale', 'best buy', 'walmart'],
        'technical': ['code', 'programming', 'software', 'python', 'javascript', 'java',
                     'error', 'fix', 'solution', 'bug', 'debug', 'api', 'github', 'stack',
                     'computer', 'tech', 'technology', 'linux', 'windows', 'mac'],
        'personal': ['my', 'me', 'self', 'personal', 'diary', 'journal', 'thoughts',
                    'vlog', 'blog', 'life', 'day in the life', 'my life'],
        'health': ['health', 'fitness', 'exercise', 'diet', 'workout', 'yoga', 'meditation',
                  'doctor', 'symptom', 'medical', 'hospital', 'medicine', 'weight loss'],
        'travel': ['travel', 'vacation', 'hotel', 'flight', 'destination', 'tour', 'trip',
                  'airplane', 'beach', 'mountain', 'city', 'country', 'passport'],
        'food': ['recipe', 'cooking', 'food', 'meal', 'restaurant', 'chef', 'baking',
                'kitchen', 'delicious', 'eating', 'dinner', 'lunch', 'breakfast'],
        'sports': ['sports', 'football', 'soccer', 'basketball', 'baseball', 'tennis',
                  'olympics', 'nba', 'nfl', 'mlb', 'game highlights', 'score'],
    }
    
    categorized = []
    for i, query in enumerate(queries):
        query_lower = query.lower()
        found_category = 'other'
        
        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                found_category = category
                break
        
        categorized.append({
            'query': query,
            'category': found_category
        })
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"  Categorized {i + 1} queries...")
    
    return pd.DataFrame(categorized)

def create_sentiment_visualizations(df_sentiment, df_categories, output_dir='.'):
    """Create comprehensive sentiment analysis visualizations"""
    print("Creating visualizations...")
    
    if df_sentiment.empty:
        print("No data to visualize!")
        return
    
    # Set style
    plt.style.use('default')
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Sentiment distribution pie chart
    ax1 = plt.subplot(2, 3, 1)
    sentiment_counts = df_sentiment['sentiment'].value_counts()
    colors = {'positive': '#4CAF50', 'neutral': '#2196F3', 'negative': '#F44336'}
    sentiment_colors = [colors.get(s, '#888888') for s in sentiment_counts.index]
    wedges, texts, autotexts = ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, 
                                       autopct='%1.1f%%', colors=sentiment_colors, startangle=90)
    ax1.set_title('Overall Sentiment Distribution', fontsize=14, fontweight='bold')
    
    # 2. Sentiment over time (line plot)
    ax2 = plt.subplot(2, 3, 2)
    # Create a copy for time series analysis
    df_time = df_sentiment.copy()
    df_time['timestamp'] = pd.to_datetime(df_time['timestamp'])
    df_time.set_index('timestamp', inplace=True)
    df_time.sort_index(inplace=True)
    
    # Convert sentiment to numeric for averaging
    sentiment_numeric = {'positive': 1, 'neutral': 0, 'negative': -1}
    df_time['sentiment_numeric'] = df_time['sentiment'].map(sentiment_numeric)
    
    # Resample by month
    monthly_sentiment = df_time['sentiment_numeric'].resample('ME').mean()
    ax2.plot(monthly_sentiment.index, monthly_sentiment.values, marker='o', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(monthly_sentiment.index, 0, monthly_sentiment.values, 
                     where=monthly_sentiment.values >= 0, color='green', alpha=0.3)
    ax2.fill_between(monthly_sentiment.index, 0, monthly_sentiment.values, 
                     where=monthly_sentiment.values < 0, color='red', alpha=0.3)
    ax2.set_title('Average Sentiment Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Sentiment Score')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Category distribution
    ax3 = plt.subplot(2, 3, 3)
    category_counts = df_categories['category'].value_counts()
    top_categories = category_counts.head(10)  # Show top 10 categories
    top_categories.plot(kind='barh', ax=ax3, color='steelblue')
    ax3.set_title('Top Search Categories', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Number of Searches')
    
    # 4. Sentiment by category
    ax4 = plt.subplot(2, 3, 4)
    # Merge sentiment and category data
    merged_df = pd.merge(df_sentiment[['query', 'sentiment']], 
                        df_categories[['query', 'category']], on='query', how='inner')
    
    if not merged_df.empty:
        # Create pivot table for top categories only
        top_cat_names = top_categories.index.tolist()
        merged_df = merged_df[merged_df['category'].isin(top_cat_names)]
        
        pivot_table = pd.crosstab(merged_df['category'], merged_df['sentiment'])
        if not pivot_table.empty:
            # Convert to percentages
            pivot_table = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100
            
            # Reorder columns for consistent color mapping
            pivot_table = pivot_table.reindex(columns=['positive', 'neutral', 'negative'], fill_value=0)
            
            pivot_table.plot(kind='bar', stacked=True, ax=ax4, 
                           color=[colors['positive'], colors['neutral'], colors['negative']])
            ax4.set_title('Sentiment by Category (%)', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Category')
            ax4.set_ylabel('Percentage')
            ax4.legend(title='Sentiment')
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax4.text(0.5, 0.5, 'No category-sentiment data', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Sentiment by Category', fontsize=14, fontweight='bold')
    
    # 5. Hourly sentiment pattern
    ax5 = plt.subplot(2, 3, 5)
    df_time['hour'] = df_time.index.hour
    hourly_sentiment = df_time.groupby('hour')['sentiment_numeric'].mean()
    ax5.bar(hourly_sentiment.index, hourly_sentiment.values, color='purple', alpha=0.7)
    ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax5.set_title('Average Sentiment by Hour of Day', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Hour of Day')
    ax5.set_ylabel('Average Sentiment')
    ax5.set_xticks(range(0, 24, 2))
    
    # 6. Word cloud of most common words
    ax6 = plt.subplot(2, 3, 6)
    
    # Combine all search queries
    all_text = ' '.join(df_sentiment['query'].astype(str))
    
    # Create word cloud
    wordcloud = WordCloud(width=400, height=300, background_color='white',
                         max_words=100, colormap='viridis').generate(all_text)
    
    ax6.imshow(wordcloud, interpolation='bilinear')
    ax6.set_title('Most Common Search Words', fontsize=14, fontweight='bold')
    ax6.axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'youtube_search_sentiment_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Main visualizations saved to: {output_path}")
    
    # Create detailed analysis plots
    create_detailed_visualizations(df_sentiment, output_dir)
    
    plt.show()

def create_detailed_visualizations(df_sentiment, output_dir):
    """Create additional detailed visualizations"""
    if df_sentiment.empty:
        return
    
    # Set up the figure
    plt.figure(figsize=(14, 10))
    
    # 1. Distribution of sentiment scores
    plt.subplot(2, 2, 1)
    if 'compound' in df_sentiment.columns:
        plt.hist(df_sentiment['compound'], bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral')
        plt.axvline(x=df_sentiment['compound'].mean(), color='green', linestyle='-', 
                   alpha=0.7, label=f'Mean: {df_sentiment["compound"].mean():.3f}')
        plt.title('Distribution of Sentiment Scores', fontsize=12, fontweight='bold')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 2. Top positive and negative searches
    plt.subplot(2, 2, 2)
    if 'compound' in df_sentiment.columns:
        # Get top 5 positive and negative
        top_positive = df_sentiment.nlargest(5, 'compound')
        top_negative = df_sentiment.nsmallest(5, 'compound')
        
        # Prepare data for plotting
        labels = []
        values = []
        colors = []
        
        for _, row in top_positive.iterrows():
            labels.append(f"Pos: {row['query'][:30]}...")
            values.append(row['compound'])
            colors.append('green')
            
        for _, row in top_negative.iterrows():
            labels.append(f"Neg: {row['query'][:30]}...")
            values.append(row['compound'])
            colors.append('red')
        
        y_pos = range(len(values))
        plt.barh(y_pos, values, color=colors, alpha=0.6)
        plt.yticks(y_pos, labels, fontsize=8)
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        plt.title('Extreme Sentiment Searches', fontsize=12, fontweight='bold')
        plt.xlabel('Sentiment Score')
        plt.grid(True, alpha=0.3, axis='x')
    
    # 3. Weekly pattern
    plt.subplot(2, 2, 3)
    df_temp = df_sentiment.copy()
    df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])
    df_temp['weekday'] = df_temp['timestamp'].dt.day_name()
    df_temp['weekday_num'] = df_temp['timestamp'].dt.dayofweek
    
    if 'compound' in df_temp.columns:
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_sentiment = df_temp.groupby('weekday')['compound'].mean().reindex(weekday_order)
        
        plt.bar(range(len(weekday_sentiment)), weekday_sentiment.values, 
               color=['blue' if x >= 0 else 'red' for x in weekday_sentiment.values])
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.title('Average Sentiment by Day of Week', fontsize=12, fontweight='bold')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Sentiment')
        plt.xticks(range(len(weekday_sentiment)), weekday_sentiment.index, rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
    
    # 4. Query length vs sentiment
    plt.subplot(2, 2, 4)
    df_temp['query_length'] = df_temp['query'].str.len()
    
    if 'compound' in df_temp.columns:
        plt.scatter(df_temp['query_length'], df_temp['compound'], 
                   alpha=0.5, s=20, c=df_temp['compound'], cmap='coolwarm')
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.title('Sentiment vs Query Length', fontsize=12, fontweight='bold')
        plt.xlabel('Query Length (characters)')
        plt.ylabel('Sentiment Score')
        plt.colorbar(label='Sentiment Score')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    detailed_path = os.path.join(output_dir, 'youtube_search_detailed_analysis.png')
    plt.savefig(detailed_path, dpi=150, bbox_inches='tight')
    print(f"✅ Detailed analysis saved to: {detailed_path}")

def generate_report(df_sentiment, df_categories, output_dir='.'):
    """Generate a comprehensive text report"""
    print("Generating report...")
    
    if df_sentiment.empty:
        print("No data for report!")
        return
    
    report = []
    report.append("=" * 60)
    report.append("YOUTUBE SEARCH HISTORY SENTIMENT ANALYSIS REPORT")
    report.append("=" * 60)
    report.append(f"\nTotal searches analyzed: {len(df_sentiment)}")
    
    # Basic statistics
    if 'timestamp' in df_sentiment.columns:
        df_sentiment['timestamp'] = pd.to_datetime(df_sentiment['timestamp'])
        date_range = f"{df_sentiment['timestamp'].min().date()} to {df_sentiment['timestamp'].max().date()}"
        report.append(f"Date range: {date_range}")
        report.append(f"Total days: {(df_sentiment['timestamp'].max() - df_sentiment['timestamp'].min()).days}")
    
    # Sentiment summary
    sentiment_counts = df_sentiment['sentiment'].value_counts()
    report.append("\n--- SENTIMENT SUMMARY ---")
    for sentiment in ['positive', 'neutral', 'negative']:
        count = sentiment_counts.get(sentiment, 0)
        percentage = (count / len(df_sentiment)) * 100
        report.append(f"{sentiment.capitalize()}: {count} searches ({percentage:.1f}%)")
    
    # Average sentiment
    if 'compound' in df_sentiment.columns:
        avg_sentiment = df_sentiment['compound'].mean()
        report.append(f"\nAverage sentiment score: {avg_sentiment:.3f}")
        
        if avg_sentiment > 0.05:
            overall = "Positive"
        elif avg_sentiment < -0.05:
            overall = "Negative"
        else:
            overall = "Neutral"
        report.append(f"Overall sentiment: {overall}")
    
    # Category summary
    report.append("\n--- CATEGORY DISTRIBUTION ---")
    category_counts = df_categories['category'].value_counts().head(10)
    for category, count in category_counts.items():
        percentage = (count / len(df_categories)) * 100
        report.append(f"{category.capitalize()}: {count} searches ({percentage:.1f}%)")
    
    # Time-based insights
    report.append("\n--- TIME-BASED INSIGHTS ---")
    if 'timestamp' in df_sentiment.columns and 'compound' in df_sentiment.columns:
        df_sentiment['hour'] = pd.to_datetime(df_sentiment['timestamp']).dt.hour
        hourly_avg = df_sentiment.groupby('hour')['compound'].mean()
        
        if not hourly_avg.empty:
            best_hour = hourly_avg.idxmax()
            worst_hour = hourly_avg.idxmin()
            report.append(f"Most positive hour: {best_hour}:00 (score: {hourly_avg.max():.3f})")
            report.append(f"Most negative hour: {worst_hour}:00 (score: {hourly_avg.min():.3f})")
    
    # Most common words
    report.append("\n--- MOST COMMON WORDS ---")
    all_text = ' '.join(df_sentiment['query'].astype(str).str.lower())
    words = re.findall(r'\b\w+\b', all_text)
    word_counts = Counter(words).most_common(10)
    
    for word, count in word_counts:
        report.append(f"{word}: {count} times")
    
    # Write report to file
    report_path = os.path.join(output_dir, 'youtube_search_sentiment_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"📄 Report saved to: {report_path}")
    
    # Print summary to console
    print("\n" + "\n".join(report[:25]))

def main():
    """Main function to run the sentiment analysis"""
    print("=" * 60)
    print("YouTube Search History Sentiment Analyzer")
    print("=" * 60)
    
    # Find HTML files in current directory
    html_files = []
    for file in os.listdir('.'):
        if file.lower().endswith('.html'):
            html_files.append(file)
    
    if not html_files:
        print("❌ No HTML files found in current directory!")
        print("Please place your YouTube HTML file in the same folder as this script.")
        print("Common file names: search-history.html, watch-history.html, history.html")
        return
    
    print(f"\nFound HTML files: {html_files}")
    
    # Let user choose if multiple files
    if len(html_files) > 1:
        print("\nMultiple HTML files found. Please select:")
        for i, file in enumerate(html_files, 1):
            print(f"{i}. {file}")
        
        try:
            choice = int(input("\nEnter number: ")) - 1
            if 0 <= choice < len(html_files):
                html_file = html_files[choice]
            else:
                html_file = html_files[0]
        except:
            html_file = html_files[0]
            print(f"Using first file: {html_file}")
    else:
        html_file = html_files[0]
    
    print(f"\n📂 Processing: {html_file}")
    
    # Extract search history
    print("\n🔍 Extracting search history...")
    search_history = extract_search_history(html_file)
    
    if not search_history:
        print("❌ No search history could be extracted.")
        print("\nPossible reasons:")
        print("1. This might be a watch-history file, not search-history")
        print("2. The HTML structure might be different")
        print("3. Try a different HTML file from your Takeout")
        return
    
    print(f"✅ Found {len(search_history)} search entries")
    
    # Convert to DataFrame and clean
    df = pd.DataFrame(search_history)
    print(f"\n🧹 Cleaning and filtering data...")
    df_clean = clean_search_dataframe(df)
    
    if df_clean.empty:
        print("❌ No valid search queries after cleaning.")
        return
    
    print(f"📊 {len(df_clean)} unique, valid search queries to analyze")
    
    # Perform sentiment analysis
    print("\n📈 Performing sentiment analysis (this may take a moment)...")
    df_sentiment = analyze_sentiment_vader(df_clean['cleaned_query'].tolist())
    
    # Add timestamp back to sentiment dataframe
    df_sentiment['timestamp'] = df_clean['timestamp'].values
    
    # Categorize searches
    print("\n🏷️  Categorizing search topics...")
    df_categories = categorize_search_topics(df_clean['cleaned_query'].tolist())
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    create_sentiment_visualizations(df_sentiment, df_categories)
    
    # Generate report
    print("\n📋 Generating report...")
    generate_report(df_sentiment, df_categories)
    
    # Save all data
    print("\n💾 Saving complete dataset...")
    
    # Save sentiment data
    sentiment_csv = 'youtube_search_sentiment_data.csv'
    df_sentiment.to_csv(sentiment_csv, index=False, encoding='utf-8')
    print(f"✅ Sentiment data saved to: {sentiment_csv}")
    
    # Save categorized data
    categories_csv = 'youtube_search_categories.csv'
    df_categories.to_csv(categories_csv, index=False, encoding='utf-8')
    print(f"✅ Category data saved to: {categories_csv}")
    
    # Save raw cleaned data
    raw_csv = 'youtube_search_cleaned.csv'
    df_clean.to_csv(raw_csv, index=False, encoding='utf-8')
    print(f"✅ Cleaned search data saved to: {raw_csv}")
    
    print("\n" + "=" * 60)
    print("Sentiment Analysis Complete! 🎉")
    print("=" * 60)
    print("\nFiles created:")
    print("1. youtube_search_sentiment_analysis.png - Main visualizations")
    print("2. youtube_search_detailed_analysis.png - Detailed charts")
    print("3. youtube_search_sentiment_report.txt - Summary report")
    print("4. youtube_search_sentiment_data.csv - Full sentiment data")
    print("5. youtube_search_categories.csv - Categorized searches")
    print("6. youtube_search_cleaned.csv - Cleaned search queries")

if __name__ == "__main__":
    # Install required packages if not already installed
    import subprocess
    import sys
    
    required_packages = ['textblob', 'nltk', 'wordcloud', 'seaborn']
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Initialize NLTK
    try:
        nltk.data.find('vader_lexicon')
    except:
        print("Downloading NLTK data...")
        nltk.download('vader_lexicon')
        nltk.download('punkt')
    
    main()