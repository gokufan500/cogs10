#!/usr/bin/env python3
"""
YouTube Watch History Analyzer
Processes YouTube watch history HTML file and creates visualizations
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator
from datetime import datetime
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def extract_watch_history(html_file_path):
    """
    Extract watch history from YouTube Takeout HTML file
    Returns: List of dictionaries with video info
    """
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Pattern to match watch history entries
    # This pattern looks for the watch history div structure
    pattern = r'<div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">([\s\S]*?)</div>'
    entries = re.findall(pattern, html_content)
    
    watch_history = []
    
    for entry in entries:
        try:
            # Clean the HTML tags and extract text
            entry_text = re.sub(r'<[^>]+>', ' ', entry)
            entry_text = re.sub(r'\s+', ' ', entry_text).strip()
            
            # Extract date and time (common pattern in YouTube takeout)
            date_pattern = r'(\w{3}\s+\d{1,2},\s+\d{4},\s+\d{1,2}:\d{2}:\d{2}\s*[AP]M\s*[A-Z]+)'
            date_match = re.search(date_pattern, entry_text)
            
            if date_match:
                date_str = date_match.group(1)
                
                # Try to parse the date
                try:
                    # Handle different date formats
                    for fmt in ['%b %d, %Y, %I:%M:%S %p %Z', 
                                '%b %d, %Y, %I:%M:%S %p GMT',
                                '%b %d, %Y, %I:%M:%S %p']:
                        try:
                            watch_time = datetime.strptime(date_str, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        continue  # Skip if no format matched
                except Exception:
                    continue
                
                # Extract video title/description (text before the date)
                video_info = entry_text.split(date_str)[0].strip()
                
                # Clean up the video info
                if 'Watched' in video_info:
                    video_info = video_info.replace('Watched', '').strip()
                
                # Look for URL pattern
                url_pattern = r'href="(https://www\.youtube\.com/watch\?v=[^"]+)"'
                url_match = re.search(url_pattern, entry)
                video_url = url_match.group(1) if url_match else None
                
                # Extract channel if present (common pattern: "by Channel Name")
                channel_match = re.search(r'by\s+([^0-9<]+?)(?:\s*Watched|\s*$)', entry_text)
                channel = channel_match.group(1).strip() if channel_match else "Unknown"
                
                watch_history.append({
                    'timestamp': watch_time,
                    'video_info': video_info[:200],  # Limit length
                    'channel': channel,
                    'url': video_url,
                    'raw_entry': entry_text[:300]  # Store first 300 chars for debug
                })
                
        except Exception as e:
            continue  # Skip problematic entries
    
    return watch_history

def extract_watch_history_alternative(html_file_path):
    """
    Alternative extraction method - looks for more flexible patterns
    """
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # More flexible pattern for different HTML structures
    # Look for entries that contain YouTube links
    pattern = r'<a[^>]*href="(https://www\.youtube\.com/watch\?v=[^"]+)"[^>]*>([^<]+)</a>.*?(\w{3}\s+\d{1,2},\s+\d{4},\s+\d{1,2}:\d{2}:\d{2}\s*[AP]M\s*[A-Z]+)'
    entries = re.findall(pattern, html_content, re.DOTALL)  # Fixed: re.DOTALL instead of re.DOTNAME
    
    watch_history = []
    
    for url, title, date_str in entries:
        try:
            # Try to parse the date
            for fmt in ['%b %d, %Y, %I:%M:%S %p %Z', 
                        '%b %d, %Y, %I:%M:%S %p GMT',
                        '%b %d, %Y, %I:%M:%S %p']:
                try:
                    watch_time = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            else:
                continue  # Skip if no format matched
            
            # Extract channel from surrounding text if possible
            channel = "Unknown"
            
            watch_history.append({
                'timestamp': watch_time,
                'video_info': title.strip(),
                'channel': channel,
                'url': url,
                'raw_entry': f"{title} - {date_str}"
            })
            
        except Exception as e:
            continue
    
    return watch_history

def extract_watch_history_simple(html_file_path):
    """
    Simple extraction that works with most YouTube Takeout formats
    """
    with open(html_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all table rows (common in YouTube Takeout)
    watch_history = []
    
    # Method 1: Look for entries with "Watched" and dates
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'Watched' in line and any(month in line for month in 
                                    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
            
            # Extract date
            date_match = re.search(r'(\w{3}\s+\d{1,2},\s+\d{4},\s+\d{1,2}:\d{2}:\d{2}\s*[AP]M)', line)
            if date_match:
                date_str = date_match.group(1)
                
                # Parse date
                for fmt in ['%b %d, %Y, %I:%M:%S %p', '%b %d, %Y, %I:%M:%S %p GMT']:
                    try:
                        watch_time = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    continue
                
                # Extract video info
                video_info = line.split('Watched')[-1].split(date_str)[0].strip()
                
                # Look for channel
                channel = "Unknown"
                channel_match = re.search(r'by\s+([^<>&"]+)', line)
                if channel_match:
                    channel = channel_match.group(1).strip()
                
                # Look for URL in this or next line
                url = None
                for j in range(max(0, i-2), min(len(lines), i+3)):
                    url_match = re.search(r'href="(https://www\.youtube\.com/watch\?v=[^"]+)"', lines[j])
                    if url_match:
                        url = url_match.group(1)
                        break
                
                watch_history.append({
                    'timestamp': watch_time,
                    'video_info': video_info,
                    'channel': channel,
                    'url': url,
                    'raw_entry': line[:200]
                })
    
    return watch_history

def create_watch_history_dataframe(watch_history):
    """Convert watch history list to pandas DataFrame"""
    if not watch_history:
        return pd.DataFrame()
    
    df = pd.DataFrame(watch_history)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df

def analyze_watch_patterns(df):
    """Perform various analyses on watch history"""
    if df.empty:
        print("No watch history data found!")
        return None
    
    print(f"\n📊 Analysis Results:")
    print(f"Total videos watched: {len(df)}")
    print(f"Time period: {df.index.min()} to {df.index.max()}")
    print(f"Total days covered: {(df.index.max() - df.index.min()).days} days")
    
    # Daily watch count
    daily_counts = df.resample('D').size()
    print(f"\nAverage videos per day: {daily_counts.mean():.2f}")
    print(f"Maximum in one day: {daily_counts.max()}")
    
    # Hourly patterns
    df['hour'] = df.index.hour
    hourly_avg = df.groupby('hour').size()
    peak_hour = hourly_avg.idxmax()
    print(f"\nMost active hour: {peak_hour}:00 ({hourly_avg.max()} videos)")
    
    # Weekly patterns
    df['weekday'] = df.index.day_name()
    weekday_counts = df.groupby('weekday').size()
    print(f"\nMost active day: {weekday_counts.idxmax()} ({weekday_counts.max()} videos)")
    
    return {
        'daily_counts': daily_counts,
        'hourly_avg': hourly_avg,
        'weekday_counts': weekday_counts
    }

def create_visualizations(df, analysis_results, output_dir='.'):
    """Create visualizations of watch history"""
    if df.empty:
        print("No data to visualize!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('YouTube Watch History Analysis', fontsize=16, fontweight='bold')
    
    # 1. Time Series - Videos per day
    ax = axes[0, 0]
    daily_counts = analysis_results['daily_counts']
    ax.plot(daily_counts.index, daily_counts.values, linewidth=1, alpha=0.7)
    ax.set_title('Videos Watched Per Day', fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Videos')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(MonthLocator(interval=3))
    
    # 2. Monthly aggregation
    ax = axes[0, 1]
    monthly_counts = df.resample('ME').size()
    ax.bar(monthly_counts.index.strftime('%Y-%m'), monthly_counts.values, alpha=0.7)
    ax.set_title('Videos Watched Per Month', fontsize=12)
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Videos')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.grid(True, alpha=0.3)
    
    # 3. Hourly pattern
    ax = axes[1, 0]
    hourly_avg = analysis_results['hourly_avg']
    ax.bar(range(len(hourly_avg)), hourly_avg.values, alpha=0.7)
    ax.set_title('Average Videos by Hour of Day', fontsize=12)
    ax.set_xlabel('Hour of Day (24h)')
    ax.set_ylabel('Number of Videos')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    
    # 4. Weekday pattern
    ax = axes[1, 1]
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts = analysis_results['weekday_counts'].reindex(weekday_order)
    ax.bar(range(len(weekday_counts)), weekday_counts.values, alpha=0.7)
    ax.set_title('Videos by Day of Week', fontsize=12)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Number of Videos')
    ax.set_xticks(range(len(weekday_counts)))
    ax.set_xticklabels(weekday_counts.index, rotation=45)
    ax.grid(True, alpha=0.3)
    
    # 5. Rolling average (7-day)
    ax = axes[2, 0]
    rolling_avg = analysis_results['daily_counts'].rolling(window=7, center=True).mean()
    ax.plot(rolling_avg.index, rolling_avg.values, linewidth=2, color='red')
    ax.set_title('7-Day Rolling Average', fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Videos per Day')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    
    # 6. Cumulative count
    ax = axes[2, 1]
    cumulative = analysis_results['daily_counts'].cumsum()
    ax.plot(cumulative.index, cumulative.values, linewidth=2, color='green')
    ax.set_title('Cumulative Videos Watched', fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Videos')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'youtube_watch_history_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualizations saved to: {output_path}")
    
    # Create a simple time series plot
    plt.figure(figsize=(12, 6))
    plt.plot(analysis_results['daily_counts'].index, analysis_results['daily_counts'].values, 
             linewidth=1, alpha=0.7, marker='o', markersize=2)
    plt.title('YouTube Videos Watched Over Time', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Videos Watched per Day', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    simple_output_path = os.path.join(output_dir, 'youtube_time_series.png')
    plt.savefig(simple_output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Time series plot saved to: {simple_output_path}")
    
    plt.show()

def find_html_files():
    """Find HTML files in current directory"""
    html_files = []
    for file in os.listdir('.'):
        if file.lower().endswith('.html') and 'watch' in file.lower():
            html_files.append(file)
    
    # Also look for generic HTML files
    if not html_files:
        html_files = [f for f in os.listdir('.') if f.lower().endswith('.html')]
    
    return html_files

def save_data_to_csv(df, output_dir='.'):
    """Save processed data to CSV for further analysis"""
    if df.empty:
        return
    
    csv_path = os.path.join(output_dir, 'youtube_watch_history_processed.csv')
    # Reset index to include timestamp as a column
    df_reset = df.reset_index()
    df_reset.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"📁 Processed data saved to: {csv_path}")
    
    # Also save summary statistics
    summary = {
        'total_videos': len(df),
        'date_range_start': df.index.min(),
        'date_range_end': df.index.max(),
        'unique_channels': df['channel'].nunique(),
        'unique_days': df.resample('D').size().count()
    }
    
    summary_path = os.path.join(output_dir, 'youtube_analysis_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, default=str, indent=2)
    
    print(f"📁 Summary saved to: {summary_path}")

def main():
    """Main function to run the analysis"""
    print("=" * 60)
    print("YouTube Watch History Analyzer")
    print("=" * 60)
    
    # Find HTML files
    html_files = find_html_files()
    
    if not html_files:
        print("❌ No HTML files found in current directory!")
        print("Please place your YouTube watch-history.html file in the same folder as this script.")
        return
    
    print(f"\nFound HTML files: {html_files}")
    
    if len(html_files) > 1:
        print(f"\nMultiple HTML files found. Using first file: {html_files[0]}")
        print("If this is not the correct file, please rename others or specify.")
    
    html_file = html_files[0]
    print(f"\n📂 Processing: {html_file}")
    
    # Try extraction methods in order
    print("🔍 Extracting watch history data...")
    
    # Method 1: Simple extraction
    watch_history = extract_watch_history_simple(html_file)
    print(f"Method 1 found {len(watch_history)} entries")
    
    # Method 2: Original extraction
    if len(watch_history) < 10:
        watch_history2 = extract_watch_history(html_file)
        print(f"Method 2 found {len(watch_history2)} entries")
        if len(watch_history2) > len(watch_history):
            watch_history = watch_history2
    
    # Method 3: Alternative extraction
    if len(watch_history) < 10:
        watch_history3 = extract_watch_history_alternative(html_file)
        print(f"Method 3 found {len(watch_history3)} entries")
        if len(watch_history3) > len(watch_history):
            watch_history = watch_history3
    
    if not watch_history:
        print("❌ No watch history data could be extracted from the HTML file.")
        print("\n📋 Debugging steps:")
        print("1. Open the HTML file in a text editor and check its structure")
        print("2. Look for patterns like 'Watched' followed by dates")
        print("3. Try running the script with this debug command:")
        print("   python -c \"import re; html=open('watch-history.html').read(); print(re.findall(r'Watched.*?\\d{4}', html)[:5] if 'Watched' in html else 'No Watched found')\"")
        return
    
    print(f"✅ Successfully extracted {len(watch_history)} watch history entries")
    
    # Create DataFrame
    df = create_watch_history_dataframe(watch_history)
    
    # Analyze patterns
    analysis_results = analyze_watch_patterns(df)
    
    if analysis_results:
        # Create visualizations
        print("\n📈 Creating visualizations...")
        create_visualizations(df, analysis_results)
        
        # Save data
        save_data_to_csv(df)
        
        # Print top channels
        if 'channel' in df.columns:
            top_channels = df['channel'].value_counts().head(10)
            print("\n🏆 Top 10 Most Watched Channels:")
            for channel, count in top_channels.items():
                print(f"  {channel}: {count} videos")
    
    print("\n" + "=" * 60)
    print("Analysis Complete! 🎉")
    print("=" * 60)

if __name__ == "__main__":
    main()