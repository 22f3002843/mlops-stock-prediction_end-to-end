import gc
import os
import pandas as pd
from feast import FeatureStore

def fetch_leak_free_training_data():
    print("Initializing Feast store...")
    store = FeatureStore(repo_path="./feature_repository")
    
    print("Reading parquet data...")
    raw_df = pd.read_parquet("feature_repository/data/stock_features.parquet")
    
    # Use 1,000 rows to keep memory footprint well under Cloud Shell limits
    subset_size = min(1000, len(raw_df))
    sample_df = raw_df.tail(subset_size).copy().reset_index(drop=True)
    
    # Clean raw_df from RAM immediately
    del raw_df
    gc.collect()
    
    entity_df = sample_df[['stock_name', 'event_timestamp', 'target']].copy()
    
    print(f"Fetching features for {len(entity_df)} records via Feast...")
    
    # Perform Point-in-Time join using Feast
    training_data = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "stock_rolling_features:rolling_avg_10",
            "stock_rolling_features:volume_sum_10",
        ]
    ).to_df()
    
    # Ensure destination directory exists and save CSV
    os.makedirs("data", exist_ok=True)
    training_data.to_csv("data/training_dataset.csv", index=False)
    
    print("\n✅ SUCCESS: Point-in-time training dataset generated at data/training_dataset.csv!")
    print("\nDataset Preview:")
    print(training_data.head())

if __name__ == "__main__":
    fetch_leak_free_training_data()