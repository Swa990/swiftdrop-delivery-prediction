
import time
import joblib
import numpy as np
import pandas as pd
import onnxruntime as ort

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

TEST_DATA_PATH = "./data/processed/test.csv"
PREPROCESSOR_PATH = "./models/preprocessor.pkl"
ONNX_PATH = "./models/xgboost_model.onnx"

TARGET = "Time_taken (min)"
N_ROWS = 1000


# --------------------------------------------------
# 2. Load test data
# --------------------------------------------------

df_test = pd.read_csv(TEST_DATA_PATH)

X_test = df_test.drop(columns=[TARGET])
y_test = df_test[TARGET].values

print("Test data shape:", df_test.shape)


# --------------------------------------------------
# 3. Load fitted preprocessor
# --------------------------------------------------

preprocessor = joblib.load(PREPROCESSOR_PATH)

X_test_processed = preprocessor.transform(X_test)

# Convert sparse matrix to dense NumPy array
if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()

print("Processed test shape:", X_test_processed.shape)


# --------------------------------------------------
# 4. Select 1000 rows
# --------------------------------------------------

X_1000 = X_test_processed[:N_ROWS].astype(np.float32)
y_1000 = y_test[:N_ROWS]


# --------------------------------------------------
# 5. Load ONNX model
# --------------------------------------------------

session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name


# --------------------------------------------------
# 6. ONNX prediction
# --------------------------------------------------

# Warm-up
session.run(
    None,
    {input_name: X_1000}
)

start_time = time.perf_counter()

outputs = session.run(
    None,
    {input_name: X_1000}
)

end_time = time.perf_counter()


# --------------------------------------------------
# 7. Calculate metrics
# --------------------------------------------------

onnx_predictions = np.asarray(outputs[0]).reshape(-1)

latency_ms = (end_time - start_time) * 1000

mae = mean_absolute_error(
    y_1000,
    onnx_predictions
)

mse = mean_squared_error(
    y_1000,
    onnx_predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_1000,
    onnx_predictions
)


# --------------------------------------------------
# 8. Print results
# --------------------------------------------------

print("\n" + "=" * 45)
print("        ONNX RUNTIME RESULTS")
print("=" * 45)

print(f"MAE              : {mae:.4f} minutes")
print(f"MSE              : {mse:.4f}")
print(f"RMSE             : {rmse:.4f} minutes")
print(f"R² Score         : {r2:.4f}")
print(f"Latency (1000)   : {latency_ms:.4f} ms")
print(f"Latency limit    : 50 ms")

if latency_ms <= 50:
    print("Status            : PASS")
else:
    print("Status            : FAIL")

print("=" * 45)
