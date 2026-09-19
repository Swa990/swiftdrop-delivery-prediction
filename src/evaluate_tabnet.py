import time
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pytorch_tabnet.tab_model import TabNetRegressor

from src.preprocessing import create_features


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "models/tabnet_best"
PREPROCESSOR_PATH = "models/tabnet_preprocessor.pkl"

TARGET = "Time_taken (min)"
LATENCY_ROWS = 1000


# --------------------------------------------------
# 2. Load test data
# --------------------------------------------------

df = pd.read_csv(TEST_PATH)

print("Test data shape:", df.shape)

X_test = df.drop(columns=[TARGET])
y_test = df[TARGET].values


# --------------------------------------------------
# 3. Feature engineering
# --------------------------------------------------

X_test = create_features(X_test)

print("Features after engineering:", X_test.shape)


# --------------------------------------------------
# 4. Load preprocessor
# --------------------------------------------------

preprocessor = joblib.load(PREPROCESSOR_PATH)

X_test_processed = preprocessor.transform(X_test)

# Convert sparse matrix if necessary
if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()

X_test_processed = np.asarray(
    X_test_processed,
    dtype=np.float32
)

print("Processed test shape:", X_test_processed.shape)


# --------------------------------------------------
# 5. Load TabNet model
# --------------------------------------------------

model = TabNetRegressor()

model.load_model(MODEL_PATH + ".zip")

print("TabNet model loaded successfully.")


# --------------------------------------------------
# 6. Prediction on complete test set
# --------------------------------------------------

y_pred = model.predict(X_test_processed).reshape(-1)


# --------------------------------------------------
# 7. Evaluation metrics
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\nTABNET TEST RESULTS")
print("-" * 40)
print(f"MAE              : {mae:.4f} minutes")
print(f"MSE              : {mse:.4f}")
print(f"RMSE             : {rmse:.4f} minutes")
print(f"R² Score         : {r2:.4f}")


# --------------------------------------------------
# 8. Latency test for 1000 rows
# --------------------------------------------------

X_latency = X_test_processed[:LATENCY_ROWS]


# Warm-up
_ = model.predict(X_latency)


start_time = time.perf_counter()

_ = model.predict(X_latency)

end_time = time.perf_counter()

latency_ms = (end_time - start_time) * 1000

mean_time_per_row = latency_ms / LATENCY_ROWS


print("\nLATENCY RESULTS")
print("-" * 40)
print(f"Rows tested       : {LATENCY_ROWS}")
print(f"Latency (1000)    : {latency_ms:.4f} ms")
print(f"Mean time / row   : {mean_time_per_row:.6f} ms")

print("\nLatency limit     : 50 ms")

if latency_ms <= 50:
    print("Status            : PASS")
else:
    print("Status            : FAIL")