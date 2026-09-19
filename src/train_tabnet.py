import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import torch

from sklearn.metrics import mean_absolute_error
from pytorch_tabnet.tab_model import TabNetRegressor

from src.preprocessing import (
    create_features,
    create_preprocessor
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

TRAIN_PATH = "data/processed/train.csv"
VALIDATION_PATH = "data/processed/validation.csv"
TEST_PATH = "data/processed/test.csv"

MODEL_PATH = "models/tabnet_best"
PREPROCESSOR_PATH = "models/tabnet_preprocessor.pkl"

TARGET = "Time_taken (min)"

RANDOM_STATE = 42

MAX_EPOCHS = 100
BATCH_SIZE = 1024
VIRTUAL_BATCH_SIZE = 128


# --------------------------------------------------
# 2. Set random seeds
# --------------------------------------------------

np.random.seed(RANDOM_STATE)


# --------------------------------------------------
# 3. Load datasets
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)
test_df = pd.read_csv(TEST_PATH)

print("\nDataset shapes")
print("----------------")
print("Train:", train_df.shape)
print("Validation:", validation_df.shape)
print("Test:", test_df.shape)


# --------------------------------------------------
# 4. Separate features and target
# --------------------------------------------------

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET].values

X_val = validation_df.drop(columns=[TARGET])
y_val = validation_df[TARGET].values

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET].values


# --------------------------------------------------
# 5. Feature engineering
# --------------------------------------------------

X_train = create_features(X_train)
X_val = create_features(X_val)
X_test = create_features(X_test)

print("\nFeatures after engineering")
print("--------------------------")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# --------------------------------------------------
# 6. Create preprocessor
# --------------------------------------------------

preprocessor = create_preprocessor(X_train)


# --------------------------------------------------
# 7. Fit ONLY on training data
# --------------------------------------------------

X_train_processed = preprocessor.fit_transform(
    X_train
)


# --------------------------------------------------
# 8. Transform validation and test
# --------------------------------------------------

X_val_processed = preprocessor.transform(
    X_val
)

X_test_processed = preprocessor.transform(
    X_test
)


# --------------------------------------------------
# 9. Convert sparse matrices to dense
# --------------------------------------------------

if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()

if hasattr(X_val_processed, "toarray"):
    X_val_processed = X_val_processed.toarray()

if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()


# --------------------------------------------------
# 10. Convert to float32
# --------------------------------------------------

X_train_processed = np.asarray(
    X_train_processed,
    dtype=np.float32
)

X_val_processed = np.asarray(
    X_val_processed,
    dtype=np.float32
)

X_test_processed = np.asarray(
    X_test_processed,
    dtype=np.float32
)


print("\nProcessed feature shapes")
print("------------------------")
print("Train:", X_train_processed.shape)
print("Validation:", X_val_processed.shape)
print("Test:", X_test_processed.shape)


# --------------------------------------------------
# 11. Check feature consistency
# --------------------------------------------------

assert (
    X_train_processed.shape[1]
    == X_val_processed.shape[1]
)

assert (
    X_train_processed.shape[1]
    == X_test_processed.shape[1]
)

print("\nFeature count is consistent.")


# --------------------------------------------------
# 12. Save preprocessor
# --------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    preprocessor,
    PREPROCESSOR_PATH
)

print(
    f"Preprocessor saved to: "
    f"{PREPROCESSOR_PATH}"
)


# --------------------------------------------------
# 13. Create TabNet model
# --------------------------------------------------

input_features = X_train_processed.shape[1]
model = TabNetRegressor(
    n_d=32,
    n_a=32,
    n_steps=5,
    gamma=1.5,
    lambda_sparse=1e-4,

    optimizer_fn=torch.optim.Adam,

    optimizer_params={
        "lr": 0.001
    },

    seed=RANDOM_STATE,

    verbose=10
)


# --------------------------------------------------
# 14. MLflow experiment
# --------------------------------------------------

mlflow.set_experiment(
    "SwiftDrop_Delivery_Prediction"
)


with mlflow.start_run(
    run_name="TabNet_Challenger"
):

    # --------------------------------------------------
    # 15. Log parameters
    # --------------------------------------------------

    mlflow.log_params({

        "model":
            "TabNetRegressor",

        "input_features":
            input_features,

        "n_d":
            32,

        "n_a":
            32,

        "n_steps":
            5,

        "gamma":
            1.5,

        "lambda_sparse":
            1e-4,

        "max_epochs":
            MAX_EPOCHS,

        "batch_size":
            BATCH_SIZE,

        "virtual_batch_size":
            VIRTUAL_BATCH_SIZE,

        "random_state":
            RANDOM_STATE
    })


    # --------------------------------------------------
    # 16. Train TabNet
    # --------------------------------------------------

    print("\nStarting TabNet training...")
    print("=" * 70)

    model.fit(

        X_train_processed,
        y_train.reshape(-1, 1),

        eval_set=[
            (
                X_train_processed,
                y_train.reshape(-1, 1)
            ),
            (
                X_val_processed,
                y_val.reshape(-1, 1)
            )
        ],

        eval_name=[
            "train",
            "validation"
        ],

        eval_metric=[
            "mae"
        ],

        max_epochs=MAX_EPOCHS,

        patience=15,

        batch_size=BATCH_SIZE,

        virtual_batch_size=VIRTUAL_BATCH_SIZE,

        num_workers=0,

        drop_last=False
    )


    # --------------------------------------------------
    # 17. Validation predictions
    # --------------------------------------------------

    val_predictions = model.predict(
        X_val_processed
    ).reshape(-1)


    # --------------------------------------------------
    # 18. Validation MAE
    # --------------------------------------------------

    val_mae = mean_absolute_error(
        y_val,
        val_predictions
    )


    print("\nTabNet Validation Results")
    print("=" * 50)

    print(
        f"Validation MAE: "
        f"{val_mae:.4f} minutes"
    )


    # --------------------------------------------------
    # 19. Log validation metric
    # --------------------------------------------------

    mlflow.log_metric(
        "validation_mae",
        val_mae
    )


    # --------------------------------------------------
    # 20. Dataset source
    # --------------------------------------------------

    mlflow.set_tag(
        "dataset_source",
        "https://www.kaggle.com/datasets/"
        "saurabhbadole/"
        "zomato-delivery-operations-analytics-dataset"
    )


# --------------------------------------------------
# 21. Save model
# --------------------------------------------------

model.save_model(
    MODEL_PATH
)


print("\nTraining completed.")
print("=" * 70)

print(
    f"Validation MAE: "
    f"{val_mae:.4f} minutes"
)

print(
    f"TabNet model saved to: "
    f"{MODEL_PATH}"
)