import json
import pandas as pd
import matplotlib.pyplot as plt

def process_search_history(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure the file is in the same folder.")
        return pd.DataFrame()

    records = []
    for entry in data:
        time_str = entry.get('time')
        if not time_str:
            continue
            
        # Check for 'From Google Ads' in the details list
        is_ad = any(d.get('name') == 'From Google Ads' for d in entry.get('details', []))
        category = "Search (Ad)" if is_ad else "Search (Organic)"
            
        records.append({
            'time': pd.to_datetime(time_str),
            'category': category
        })

    df = pd.DataFrame(records)
    
    if not df.empty:
        # Filter for data from 2025 onwards
        df = df[df['time'] >= '2025-01-01']
    return df

def plot_and_save_searches(df):
    if df.empty:
        print("No search data found for the period after 2024.")
        return

    # Prepare data for plotting
    df.set_index('time', inplace=True)
    plot_df = df.groupby([pd.Grouper(freq='D'), 'category']).size().unstack(fill_value=0)

    # Visualization
    plt.figure(figsize=(14, 7))
    
    # Define colors
    colors = {'Search (Organic)': '#2ca02c', 'Search (Ad)': '#ff7f0e'}

    for column in plot_df.columns:
        plt.plot(plot_df.index, plot_df[column], label=column, 
                 color=colors.get(column, 'gray'), linewidth=2)
        
        #

    plt.title('YouTube Search Frequency (Post-2024)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Searches', fontsize=12)
    plt.legend(frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save and Show
    output_name = "youtube_search_history.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"Graph saved as: {output_name}")
    
    plt.tight_layout()
    plt.show()

# Execution
search_df = process_search_history('search-history.json')
plot_and_save_searches(search_df)

def calculate_youtube_stats(watch_file, search_file):
    def get_stats(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            return 0, 0
        
        total_organic = 0
        total_ads = 0
        
        for entry in data:
            # Filter for post-2024 (2025 onwards)
            time = entry.get('time', '')
            if not time or time < '2025-01-01':
                continue
                
            is_ad = any(d.get('name') == 'From Google Ads' for d in entry.get('details', []))
            
            if is_ad:
                total_ads += 1
            else:
                total_organic += 1
                
        return total_organic, total_ads

    # Get data for both categories
    watch_org, watch_ads = get_stats(watch_file)
    search_org, search_ads = get_stats(search_file)
    
    # Calculate Totals
    total_items = watch_org + watch_ads + search_org + search_ads
    total_ads = watch_ads + search_ads
    
    ad_percentage = (total_ads / total_items * 100) if total_items > 0 else 0

    # Print Results
    print("--- YouTube History Summary (Post-2024) ---")
    print(f"{'Category':<15} | {'Organic':<10} | {'Ads':<10}")
    print("-" * 40)
    print(f"{'Videos Watched':<15} | {watch_org:<10} | {watch_ads:<10}")
    print(f"{'Searches Made':<15} | {search_org:<10} | {search_ads:<10}")
    print("-" * 40)
    print(f"Total Interactions: {total_items}")
    print(f"Total Ad Content:   {total_ads}")
    print(f"Ad Percentage:      {ad_percentage:.2f}%")

# Usage
calculate_youtube_stats('watch-history.json', 'search-history.json')
