from datetime import timedelta
from feast import (
    Entity,
    Field,
    FeatureView,
    FileSource,
    ValueType,
)
from feast.types import Float64, String, Int64

# Define entity
stock = Entity(
    name="stock_name",
    value_type=ValueType.STRING,
    description="Name identifier of the stock ticker",
)

# File source pointing to relative path from feature_repository
file_source = FileSource(
    name="stock_stats_source",
    path="data/stock_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define Feature View
stock_rolling_features = FeatureView(
    name="stock_rolling_features",
    entities=[stock],
    ttl=timedelta(days=365*10),
    schema=[
        Field(name="rolling_avg_10", dtype=Float64),
        Field(name="volume_sum_10", dtype=Float64),
    ],
    online=True,
    source=file_source,
)