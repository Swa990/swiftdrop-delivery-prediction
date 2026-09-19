import joblib
import numpy as np
import pandas as pd
import onnxruntime as ort

from sklearn.metrics import mean_absolute_error


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

TEST_DATA_PATH = "./data/processed/test.csv"
PIPELINE_PATH = "./models/xgboost_pipeline.pkl"
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

X_1000 = X_test.iloc[:N_ROWS]
y_1000 = y_test[:N_ROWS]

print("Test rows:", len(X_1000))


# --------------------------------------------------
# 3. Load complete native pipeline
# --------------------------------------------------

pipeline = joblib.load(PIPELINE_PATH)

print("Native pipeline loaded.")


# --------------------------------------------------
# 4. Native predictions from complete pipeline
# --------------------------------------------------

native_predictions = pipeline.predict(X_1000)


# --------------------------------------------------
# 5. Load preprocessor
# --------------------------------------------------

preprocessor = joblib.load(PREPROCESSOR_PATH)

X_processed = preprocessor.transform(X_1000)

if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()

X_processed = X_processed.astype(np.float32)

print("Processed shape:", X_processed.shape)


# --------------------------------------------------
# 6. Extract native XGBoost model
# --------------------------------------------------

xgb_model = pipeline.named_steps["model"]

xgb_predictions = xgb_model.predict(X_processed)


# --------------------------------------------------
# 7. ONNX Runtime
# --------------------------------------------------

session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

outputs = session.run(
    None,
    {
        input_name: X_processed
    }
)

onnx_predictions = np.asarray(
    outputs[0]
).reshape(-1)


# --------------------------------------------------
# 8. Compare predictions
# --------------------------------------------------

print("\n" + "=" * 60)
print("PREDICTION COMPARISON")
print("=" * 60)

print("\nFirst 10 predictions:")

comparison = pd.DataFrame({
    "Actual": y_1000[:10],
    "Pipeline": native_predictions[:10],
    "XGBoost": xgb_predictions[:10],
    "ONNX": onnx_predictions[:10]
})

print(comparison)


# --------------------------------------------------
# 9. Compare native pipeline vs XGBoost
# --------------------------------------------------

pipeline_difference = np.abs(
    native_predictions - xgb_predictions
)

print("\nPipeline vs XGBoost")
print("-------------------")
print(
    "Maximum difference:",
    pipeline_difference.max()
)


# --------------------------------------------------
# 10. Compare XGBoost vs ONNX
# --------------------------------------------------

onnx_difference = np.abs(
    xgb_predictions - onnx_predictions
)

print("\nXGBoost vs ONNX")
print("----------------")
print(
    "Maximum difference:",
    onnx_difference.max()
)

print(
    "Mean difference:",
    onnx_difference.mean()
)


# --------------------------------------------------
# 11. MAE comparison
# --------------------------------------------------

native_mae = mean_absolute_error(
    y_1000,
    xgb_predictions
)

onnx_mae = mean_absolute_error(
    y_1000,
    onnx_predictions
)

print("\n" + "=" * 60)
print("MAE COMPARISON")
print("=" * 60)

print(f"Native XGBoost MAE : {native_mae:.4f}")
print(f"ONNX MAE           : {onnx_mae:.4f}")


# import joblib


# PIPELINE_PATH = "models/xgboost_pipeline.pkl"
# PREPROCESSOR_PATH = "models/preprocessor.pkl"


# # Load complete pipeline
# pipeline = joblib.load(PIPELINE_PATH)

# # Extract preprocessor from pipeline
# pipeline_preprocessor = pipeline.named_steps["preprocessor"]


# # Load separately saved preprocessor
# saved_preprocessor = joblib.load(
#     PREPROCESSOR_PATH
# )


# print("Pipeline preprocessor:")
# print(pipeline_preprocessor)

# print("\nSeparate preprocessor:")
# print(saved_preprocessor)