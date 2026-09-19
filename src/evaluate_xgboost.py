import time
import joblib
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.data_loader import load_data


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

TEST_DATA_PATH = "./data/processed/test.csv"
MODEL_PATH = "./models/xgboost_pipeline.pkl"

TARGET = "Time_taken (min)"


# --------------------------------------------------
# 2. Load saved test data
# --------------------------------------------------

test_data = load_data(TEST_DATA_PATH)

X_test = test_data.drop(columns=[TARGET])
y_test = test_data[TARGET]

print("Test data shape:", test_data.shape)


# --------------------------------------------------
# 3. Load trained XGBoost pipeline
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# --------------------------------------------------
# 4. Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 5. Calculate MAE
# --------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)


# --------------------------------------------------
# 6. Calculate RMSE
# --------------------------------------------------

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)


# --------------------------------------------------
# 7. Calculate R² Score
# --------------------------------------------------

r2 = r2_score(
    y_test,
    y_pred
)


# --------------------------------------------------
# 8. Measure inference latency for 1000 rows
# --------------------------------------------------

X_latency = X_test.iloc[:1000]

start_time = time.perf_counter()

model.predict(X_latency)

end_time = time.perf_counter()

latency_ms = (end_time - start_time) * 1000


# --------------------------------------------------
# 9. Display results
# --------------------------------------------------

print("\nXGBoost Evaluation")
print("=" * 40)

print(f"MAE: {mae:.4f} minutes")
print(f"RMSE: {rmse:.4f} minutes")
print(f"R² Score: {r2:.4f}")
print(f"Latency for 1000 rows: {latency_ms:.4f} ms")