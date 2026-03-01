TARGET = "sales"

# Time column
TIME_COLUMN = "date"

# Categorical features available in current feature set
CATEGORICAL_FEATURES = [
    "item_id",
    "store_id",
    "event_name_1",
    "event_type_1",
]

# Lag features
LAG_FEATURES = [
    "lag_7",
    "lag_28",
]

# Rolling features
ROLLING_FEATURES = [
    "rolling_mean_7",
    "rolling_std_28",
]

# Price feature
PRICE_FEATURES = [
    "sell_price",
]

# Final feature list
FEATURE_COLUMNS = (
    CATEGORICAL_FEATURES
    + LAG_FEATURES
    + ROLLING_FEATURES
    + PRICE_FEATURES
)