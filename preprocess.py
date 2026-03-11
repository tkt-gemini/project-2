import pandas as pd
import glob
import os
import re
from pathlib import Path

# Define mental health subreddits
MENTAL_HEALTH_SUBREDDITS = {
    'depression': 'depression',
    'anxiety': 'anxiety', 
    'suicidewatch': 'suicide',
    'bipolarreddit': 'bipolar',
    'ptsd': 'ptsd',
    'autism': 'autism',
    'schizophrenia': 'schizophrenia',
    'bpd': 'bpd',
    'adhd': 'adhd',
    'edanonymous': 'eating_disorder',
    'alcoholism': 'alcoholism',
    'addiction': 'addiction',
    'socialanxiety': 'social_anxiety',
    'healthanxiety': 'health_anxiety',
    'lonely': 'lonely',
    'mentalhealth': 'mental_health_general'
}

CONTROL_SUBREDDITS = {
    'personalfinance': 'control',
    'relationships': 'control',
    'parenting': 'control',
    'teaching': 'control',
    'legaladvice': 'control',
    'jokes': 'control',
    'fitness': 'control',
    'guns': 'control',
    'meditation': 'control',
    'divorce': 'control',
    'conspiracy': 'control'
}

def extract_metadata_from_filename(filename):
    """Extract subreddit and timeframe from filename"""
    basename = os.path.basename(filename)
    parts = basename.replace('_features_tfidf_256.csv', '').split('_')
    
    # Handle multi-word subreddits
    if len(parts) >= 3:
        # Check if first two parts form a known subreddit
        potential_subreddit = f"{parts[0]}_{parts[1]}"
        if potential_subreddit in MENTAL_HEALTH_SUBREDDITS or potential_subreddit in CONTROL_SUBREDDITS:
            subreddit = potential_subreddit
            timeframe = parts[2]
        else:
            subreddit = parts[0]
            timeframe = parts[1]
    else:
        subreddit = parts[0]
        timeframe = parts[1] if len(parts) > 1 else 'unknown'
    
    return subreddit, timeframe

def assign_label(subreddit):
    """Assign label based on subreddit"""
    if subreddit in MENTAL_HEALTH_SUBREDDITS:
        return MENTAL_HEALTH_SUBREDDITS[subreddit]
    elif subreddit in CONTROL_SUBREDDITS:
        return 'control'
    else:
        return 'unknown'

def process_all_files(data_dir, output_file='merged_dataset.csv'):
    """Process and merge all CSV files"""
    
    all_data = []
    files_processed = 0
    
    # Get all CSV files
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    print(f"Found {len(csv_files)} CSV files to process")
    
    for file_path in csv_files:
        try:
            # Extract metadata
            subreddit, timeframe = extract_metadata_from_filename(file_path)
            label = assign_label(subreddit)
            
            if label == 'unknown':
                print(f"⚠️  Skipping unknown subreddit: {subreddit}")
                continue
            
            # Read CSV
            df = pd.read_csv(file_path)
            
            # Add metadata columns
            df['subreddit'] = subreddit
            df['timeframe'] = timeframe
            df['label'] = label
            df['source_file'] = os.path.basename(file_path)
            
            all_data.append(df)
            files_processed += 1
            
            if files_processed % 10 == 0:
                print(f"Processed {files_processed}/{len(csv_files)} files...")
                
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            continue
    
    # Merge all dataframes
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        
        # Save to CSV
        merged_df.to_csv(output_file, index=False)
        
        print(f"\n✅ Successfully merged {files_processed} files")
        print(f"📊 Total rows: {len(merged_df):,}")
        print(f"📊 Total columns: {len(merged_df.columns)}")
        print(f"💾 Saved to: {output_file}")
        
        # Show label distribution
        print("\n📈 Label distribution:")
        print(merged_df['label'].value_counts())
        
        # Show timeframe distribution
        print("\n⏰ Timeframe distribution:")
        print(merged_df['timeframe'].value_counts())
        
        return merged_df
    else:
        print("❌ No data to merge!")
        return None

# Run preprocessing
if __name__ == "__main__":
    data_directory = r"/home/gemini/personal/project/project-2/data/reddit-mental-health"
    output_path = r"/home/gemini/personal/project/project-2/data/reddit-mental-health-dataset.csv"
    
    dataset = process_all_files(data_directory, output_path)