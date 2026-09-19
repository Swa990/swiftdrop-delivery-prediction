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

from src.preprocessing import create_features


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

TEST_PATH = "data/processed/test.csv"

PREPROCESSOR_PATH = "models/tabnet_preprocessor.pkl"

ONNX_PATH = "models/tabnet_model.onnx"

TARGET = "Time_taken (min)"

LATENCY_ROWS = 1000


# --------------------------------------------------
# 2. Load test data
# --------------------------------------------------

print("Loading test data...")

df = pd.read_csv(TEST_PATH)

X_test = df.drop(columns=[TARGET])

y_test = df[TARGET].values

print("Test data shape:", df.shape)


# --------------------------------------------------
# 3. Feature engineering
# --------------------------------------------------

X_test = create_features(X_test)

print("Features after engineering:", X_test.shape)


# --------------------------------------------------
# 4. Load fitted preprocessor
# --------------------------------------------------

print("Loading preprocessor...")

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)

X_test_processed = preprocessor.transform(
    X_test
)

if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()

X_test_processed = np.asarray(
    X_test_processed,
    dtype=np.float32
)

print(
    "Processed test shape:",
    X_test_processed.shape
)


# --------------------------------------------------
# 5. Select 1000 rows
# --------------------------------------------------

X_latency = X_test_processed[
    :LATENCY_ROWS
]

y_latency = y_test[
    :LATENCY_ROWS
]

print(
    "Evaluation shape:",
    X_latency.shape
)


# --------------------------------------------------
# 6. Load ONNX model
# --------------------------------------------------

print("\nLoading TabNet ONNX model...")

session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CPUExecutionProvider"]
)

print(
    "ONNX model loaded successfully."
)


# --------------------------------------------------
# 7. Get input/output names
# --------------------------------------------------

input_name = session.get_inputs()[0].name

output_name = session.get_outputs()[0].name

print("Input name :", input_name)

print("Output name:", output_name)


# --------------------------------------------------
# 8. ONNX prediction
# --------------------------------------------------

print("\nRunning ONNX prediction...")

y_pred = session.run(
    [output_name],
    {
        input_name: X_latency
    }
)[0]

y_pred = np.asarray(
    y_pred
).reshape(-1)


# --------------------------------------------------
# 9. Calculate metrics
# --------------------------------------------------

mae = mean_absolute_error(
    y_latency,
    y_pred
)

mse = mean_squared_error(
    y_latency,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_latency,
    y_pred
)


# --------------------------------------------------
# 10. Print evaluation results
# --------------------------------------------------

print("\nTABNET ONNX RESULTS")
print("=" * 50)

print(
    f"Rows evaluated     : {LATENCY_ROWS}"
)

print(
    f"MAE                : {mae:.4f} minutes"
)

print(
    f"MSE                : {mse:.4f}"
)

print(
    f"RMSE               : {rmse:.4f} minutes"
)

print(
    f"R² Score           : {r2:.4f}"
)


# --------------------------------------------------
# 11. Latency benchmark
# --------------------------------------------------

print("\nRunning latency benchmark...")


# Warm-up
for _ in range(5):

    session.run(
        [output_name],
        {
            input_name: X_latency
        }
    )


# Actual measurement
start_time = time.perf_counter()

session.run(
    [output_name],
    {
        input_name: X_latency
    }
)

end_time = time.perf_counter()


latency_ms = (
    end_time - start_time
) * 1000


mean_time_per_row = (
    latency_ms / LATENCY_ROWS
)


# --------------------------------------------------
# 12. Latency results
# --------------------------------------------------

print("\nLATENCY RESULTS")
print("=" * 50)

print(
    f"Rows tested        : {LATENCY_ROWS}"
)

print(
    f"Latency (1000)     : {latency_ms:.4f} ms"
)

print(
    f"Mean time / row    : {mean_time_per_row:.6f} ms"
)

print(
    "\nLatency limit      : 50 ms"
)


if latency_ms <= 50:

    print(
        "Status             : PASS"
    )

else:

    print(
        "Status             : FAIL"
    )