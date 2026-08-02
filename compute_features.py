import os
import glob
import pandas as pd
import numpy as np

def process_stock_data(data_dir, output_parquet_path):
    # Find all CSV files in the target data directory
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    all_dfs = []
    
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        
        # Standardize column names to lowercase
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Ensure timestamp column exists and convert to datetime
        time_col = [c for c in df.columns if 'time' in c or 'date' in c][0]
        df['event_timestamp'] = pd.to_datetime(df[time_col])
        
        # Extract stock name if not present
        if 'stock_name' not in df.columns:
            stock_name = os.path.basename(file_path).split('_')[0]
            df['stock_name'] = stock_name
            
        # Sort chronologically for accurate rolling windows (Prompt requirement)
        df = df.sort_values('event_timestamp').reset_index(drop=True)
        
        # 1. Feature: rolling_avg_10 (Moving average of close price in last 10 mins)
        df['rolling_avg_10'] = df['close'].rolling(window=10, min_periods=1).mean()
        
        # 2. Feature: volume_sum_10 (Total volume traded in last 10 mins)
        vol_col = [c for c in df.columns if 'vol' in c][0]
        df['volume_sum_10'] = df[vol_col].rolling(window=10, min_periods=1).sum()
        
        # 3. Target: Predict 1 if close price 5 mins later > current close price, else 0
        df['close_future_5m'] = df['close'].shift(-5)
        df['target'] = (df['close_future_5m'] > df['close']).astype(int)
        
        # Drop rows where target cannot be computed (last 5 mins)
        df = df.dropna(subset=['target'])
        
        all_dfs.append(df)
        
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Feast requires a created timestamp column for versioning
    final_df['created_timestamp'] = pd.to_datetime('now', utc=True)
    
    # Ensure event_timestamp is timezone-aware (UTC) as required by Feast
    if final_df['event_timestamp'].dt.tz is None:
        final_df['event_timestamp'] = final_df['event_timestamp'].dt.tz_localize('UTC')
    else:
        final_df['event_timestamp'] = final_df['event_timestamp'].dt.tz_convert('UTC')
        
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    final_df.to_parquet(output_parquet_path, index=False)
    print(f"Successfully processed {len(final_df)} records saved to {output_parquet_path}")

if __name__ == "__main__":
    # Process v0 data first
    process_stock_data("data/v0", "feature_repository/data/stock_features.parquet")