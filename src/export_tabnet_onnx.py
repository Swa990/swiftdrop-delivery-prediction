import os
import joblib
import numpy as np
import pandas as pd
import torch
import onnx

from pytorch_tabnet.tab_model import TabNetRegressor
from src.preprocessing import create_features


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

TEST_PATH = "data/processed/test.csv"

MODEL_PATH = "models/tabnet_best"
PREPROCESSOR_PATH = "models/tabnet_preprocessor.pkl"

ONNX_PATH = "models/tabnet_model.onnx"

TARGET = "Time_taken (min)"

EXPORT_ROWS = 1000


# --------------------------------------------------
# 2. Load test data
# --------------------------------------------------

print("Loading test data...")

df = pd.read_csv(TEST_PATH)

X_test = df.drop(columns=[TARGET])

X_test = create_features(X_test)

print("Features after engineering:", X_test.shape)


# --------------------------------------------------
# 3. Load fitted preprocessor
# --------------------------------------------------

print("Loading TabNet preprocessor...")

preprocessor = joblib.load(PREPROCESSOR_PATH)

X_test_processed = preprocessor.transform(X_test)

if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()

X_test_processed = np.asarray(
    X_test_processed,
    dtype=np.float32
)

print("Processed test shape:", X_test_processed.shape)


# --------------------------------------------------
# 4. Use exactly 1000 rows for ONNX export
# --------------------------------------------------

X_export = X_test_processed[:EXPORT_ROWS]

print("Export input shape:", X_export.shape)


# --------------------------------------------------
# 5. Load TabNet directly on CPU
# --------------------------------------------------

print("\nLoading trained TabNet model on CPU...")

tabnet = TabNetRegressor(
    device_name="cpu"
)

tabnet.load_model(
    MODEL_PATH + ".zip"
)

print("TabNet model loaded successfully.")

print("TabNet device:", tabnet.device)


# --------------------------------------------------
# 6. Access network
# --------------------------------------------------

network = tabnet.network

network.eval()

# Explicitly move everything to CPU
network.cpu()

print("TabNet network moved to CPU.")


# --------------------------------------------------
# 7. Wrapper
# --------------------------------------------------

class TabNetONNXWrapper(torch.nn.Module):

    def __init__(self, network):
        super().__init__()

        self.network = network

    def forward(self, x):

        output, _ = self.network(x)

        return output


wrapped_model = TabNetONNXWrapper(
    network
)

wrapped_model.eval()


# --------------------------------------------------
# 8. Dummy input
# --------------------------------------------------

dummy_input = torch.from_numpy(
    X_export
).float()

print(
    "Dummy input shape:",
    dummy_input.shape
)

print(
    "Dummy input device:",
    dummy_input.device
)


# --------------------------------------------------
# 9. Export to ONNX
# --------------------------------------------------

print("\nConverting TabNet to ONNX...")

with torch.no_grad():

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        ONNX_PATH,

        export_params=True,

        opset_version=17,

        do_constant_folding=True,

        input_names=[
            "float_input"
        ],

        output_names=[
            "prediction"
        ],

        # Fixed batch size of 1000.
        # This avoids TabNet's virtual-batch tracing issue.
        dynamic_axes=None
    )


print(
    f"ONNX model saved to: {ONNX_PATH}"
)


# --------------------------------------------------
# 10. Validate ONNX
# --------------------------------------------------

print("\nValidating ONNX model...")

onnx_model = onnx.load(
    ONNX_PATH
)

onnx.checker.check_model(
    onnx_model
)

print(
    "ONNX model validation successful!"
)