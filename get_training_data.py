import pandas as pd
from feast import FeatureStore

def fetch_leak_free_training_data():
    print("Initializing Feast store...")
    store = FeatureStore(repo_path="./feature_repository")
    
    print("Reading sample parquet data...")
    raw_df = pd.read_parquet("feature_repository/data/stock_features.parquet")
    
    entity_df = raw_df[['stock_name', 'event_timestamp', 'target']].copy()
    
    print(f"Fetching features for {len(entity_df)} records via Feast...")
    
    # Perform Point-in-Time join using Feast
    training_data = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "stock_rolling_features:rolling_avg_10",
            "stock_rolling_features:volume_sum_10",
        ]
    ).to_df()
    
    # Save training dataset
    training_data.to_csv("data/training_dataset.csv", index=False)
    print("✅ Successfully generated data/training_dataset.csv!")
    print(training_data.head())

if __name__ == "__main__":
    fetch_leak_free_training_data()