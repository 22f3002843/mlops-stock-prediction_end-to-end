import os
import sys
import glob
import pandas as pd

def process_stock_data(iteration="v0"):
    if iteration == "v0":
        data_dirs = ["data/v0"]
    elif iteration == "v0_v1":
        data_dirs = ["data/v0", "data/v1"]
    else:
        raise ValueError("Invalid iteration. Use 'v0' or 'v0_v1'")

    csv_files = []
    for d in data_dirs:
        csv_files.extend(glob.glob(os.path.join(d, "*.csv")))
        
    print(f"Processing {len(csv_files)} stock CSV files for iteration: {iteration}...")
    all_dfs = []
    
    for file_path in csv_files:
        df = pd.read_csv(file_path, nrows=2500)
        df.columns = [c.strip().lower() for c in df.columns]
        
        time_col = [c for c in df.columns if 'time' in c or 'date' in c][0]
        df['event_timestamp'] = pd.to_datetime(df[time_col])
        
        if 'stock_name' not in df.columns:
            stock_name = os.path.basename(file_path).split('_')[0]
            df['stock_name'] = stock_name
            
        df = df.sort_values('event_timestamp').reset_index(drop=True)
        
        # Feature Engineering
        df['rolling_avg_10'] = df['close'].rolling(window=10, min_periods=1).mean()
        vol_col = [c for c in df.columns if 'vol' in c][0]
        df['volume_sum_10'] = df[vol_col].rolling(window=10, min_periods=1).sum()
        
        # Target Generation
        df['close_future_5m'] = df['close'].shift(-5)
        df['target'] = (df['close_future_5m'] > df['close']).astype(int)
        
        df = df.dropna(subset=['target'])
        all_dfs.append(df)
        
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df['created_timestamp'] = pd.to_datetime('now', utc=True)
    
    if final_df['event_timestamp'].dt.tz is None:
        final_df['event_timestamp'] = final_df['event_timestamp'].dt.tz_localize('UTC')
    else:
        final_df['event_timestamp'] = final_df['event_timestamp'].dt.tz_convert('UTC')
        
    output_parquet_path = "feature_repository/data/stock_features.parquet"
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    final_df.to_parquet(output_parquet_path, index=False)
    print(f"✅ Generated parquet with {len(final_df)} records at {output_parquet_path}")

if __name__ == "__main__":
    iter_choice = sys.argv[1] if len(sys.argv) > 1 else "v0"
    process_stock_data(iter_choice)