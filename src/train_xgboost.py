import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from src.data_loader import load_data
# from src.preprocessing import create_preprocessor
from src.preprocessing import create_features, create_preprocessor


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_PATH = (
    "/home/geogo/swati_workspace/Document_classifier/"
    "swiftdrop-delivery-prediction/data/raw/Zomato Dataset.csv"
)

MODEL_PATH = "models/xgboost_pipeline.pkl"

TRAIN_DATA_PATH = "data/processed/train_pool.csv"
TEST_DATA_PATH = "data/processed/test.csv"

TARGET = "Time_taken (min)"

DATASET_URL = (
    "https://www.kaggle.com/datasets/saurabhbadole/"
    "zomato-delivery-operations-analytics-dataset"
)


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

df = load_data(DATA_PATH)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 3. Separate features and target
# --------------------------------------------------

# X = df.drop(columns=[TARGET])
# y = df[TARGET]


# --------------------------------------------------
# 3. Separate features and target
# --------------------------------------------------

X = df.drop(columns=[TARGET])
y = df[TARGET]


# --------------------------------------------------
# 4. Create engineered features
# --------------------------------------------------

X = create_features(X)

print("Features after engineering:")
print(X.columns.tolist())
print("Feature shape:", X.shape)
# --------------------------------------------------
# 4. Train/Test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training pool:", X_train.shape)
print("Testing data:", X_test.shape)


# --------------------------------------------------
# 5. Save train pool and test set
# --------------------------------------------------

os.makedirs("data/processed", exist_ok=True)

train_data = X_train.copy()
train_data[TARGET] = y_train

train_data.to_csv(
    TRAIN_DATA_PATH,
    index=False
)

test_data = X_test.copy()
test_data[TARGET] = y_test

test_data.to_csv(
    TEST_DATA_PATH,
    index=False
)

print(f"Training pool saved to: {TRAIN_DATA_PATH}")
print(f"Test data saved to: {TEST_DATA_PATH}")


# --------------------------------------------------
# 6. Create preprocessor
# --------------------------------------------------

preprocessor = create_preprocessor(X_train)


# --------------------------------------------------
# 7. Create XGBoost model
# --------------------------------------------------

xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# 8. Create complete ML Pipeline
# --------------------------------------------------

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", xgb_model)
])


# --------------------------------------------------
# 9. MLflow experiment
# --------------------------------------------------

mlflow.set_experiment(
    "SwiftDrop_Delivery_Prediction"
)


with mlflow.start_run(
    run_name="XGBoost_Baseline"
):

    # --------------------------------------------------
    # 10. Train model
    # --------------------------------------------------

    model_pipeline.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------
    # 11. Make predictions
    # --------------------------------------------------

    y_pred = model_pipeline.predict(
        X_test
    )

    # --------------------------------------------------
    # 12. Calculate MAE
    # --------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    print("\nXGBoost Baseline Results")
    print("------------------------")
    print(f"MAE: {mae:.4f} minutes")

    # --------------------------------------------------
    # 13. Log parameters
    # --------------------------------------------------

    mlflow.log_params({
        "model": "XGBRegressor",
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "test_size": 0.20,
        "random_state": 42
    })

    # --------------------------------------------------
    # 14. Log dataset source
    # --------------------------------------------------

    mlflow.set_tag(
        "dataset_source",
        DATASET_URL
    )

    # --------------------------------------------------
    # 15. Log test MAE
    # --------------------------------------------------

    mlflow.log_metric(
        "test_mae",
        mae
    )

    # --------------------------------------------------
    # 16. Log model to MLflow
    # --------------------------------------------------

    mlflow.sklearn.log_model(
        model_pipeline,
        name="xgboost_pipeline",
        skops_trusted_types=[
            "numpy.dtype",
            "xgboost.core.Booster",
            "xgboost.sklearn.XGBRegressor"
        ]
    )


# --------------------------------------------------
# 17. Save model locally
# --------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model_pipeline,
    MODEL_PATH
)

print(
    f"\nModel saved to: {MODEL_PATH}"
)

