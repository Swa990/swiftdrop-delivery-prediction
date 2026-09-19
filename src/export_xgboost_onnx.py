import os
import joblib
import onnx
import onnxmltools

from onnxmltools.convert.common.data_types import FloatTensorType


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PIPELINE_PATH = "models/xgboost_pipeline.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
ONNX_PATH = "models/xgboost_model.onnx"


# --------------------------------------------------
# 2. Load trained pipeline
# --------------------------------------------------

print("Loading trained XGBoost pipeline...")

pipeline = joblib.load(PIPELINE_PATH)

preprocessor = pipeline.named_steps["preprocessor"]
xgb_model = pipeline.named_steps["model"]

print("Pipeline loaded successfully.")


# --------------------------------------------------
# 3. Save fitted preprocessor
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    preprocessor,
    PREPROCESSOR_PATH
)

print(f"Preprocessor saved to: {PREPROCESSOR_PATH}")


# --------------------------------------------------
# 4. Get number of processed features
# --------------------------------------------------

n_features = xgb_model.n_features_in_

print("Number of processed features:", n_features)


# --------------------------------------------------
# 5. Define ONNX input
# --------------------------------------------------

initial_type = [
    (
        "float_input",
        FloatTensorType([None, n_features])
    )
]


# --------------------------------------------------
# 6. Convert XGBoost model to ONNX
# --------------------------------------------------

print("Converting XGBoost model to ONNX...")

onnx_model = onnxmltools.convert_xgboost(
    xgb_model,
    initial_types=initial_type,
    target_opset=15,
    name="XGBoostDeliveryPrediction"
)


# --------------------------------------------------
# 7. Save ONNX model
# --------------------------------------------------

onnxmltools.utils.save_model(
    onnx_model,
    ONNX_PATH
)

print(f"ONNX model saved to: {ONNX_PATH}")


# --------------------------------------------------
# 8. Validate ONNX model
# --------------------------------------------------

onnx.checker.check_model(onnx_model)

print("ONNX model validation successful!")