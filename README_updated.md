# SwiftDrop Delivery Prediction Pipeline

An end-to-end machine learning pipeline for predicting food delivery time using the **Zomato Delivery Operations Analytics Dataset**. The project compares **XGBoost** with **TabNet**, tracks experiments using MLflow, exports models to ONNX, and benchmarks inference latency against a **50 ms business requirement**.

## 🚀 Live Demo

**Test the deployed Streamlit application:**  
https://swiftdrop-delivery-prediction-qnypkftgrupfgjn3tazyrm.streamlit.app/

The application is publicly accessible and can be tested directly in a browser. No local setup is required to try the prediction interface.

> **How to test:** Enter delivery-person details, weather/traffic information, order and pickup times, and restaurant/delivery coordinates, then click the prediction button to receive the estimated delivery time.

---

## 1. Problem Statement

SwiftDrop requires a machine learning system that predicts the expected delivery time of an order in minutes.

The dataset contains information about:

- Delivery personnel
- Restaurant and delivery locations
- Weather
- Traffic
- Vehicle condition
- Order type
- Festival information
- City
- Order and pickup times

**Target variable:**

```text
Time_taken (min)
```

The project evaluates two approaches:

1. **XGBoost** – traditional gradient-boosting model for tabular data
2. **TabNet** – neural architecture designed for tabular data

Models are evaluated using:

- MAE
- MSE
- RMSE
- R²
- Inference latency for 1,000 rows
- ONNX Runtime performance

**Business latency requirement:**

```text
Maximum latency for 1,000 rows: 50 ms
```

---

## 2. Dataset

### Dataset

**Zomato Delivery Operations Analytics Dataset**

The dataset contains approximately **45,584 records and 20 original columns**.

The target is:

```text
Time_taken (min)
```

This is a continuous numerical target, so the task is a **regression problem**.

### Dataset storage

The raw and processed datasets are **not committed to GitHub**. They are stored externally to keep the repository lightweight and avoid committing data files.

**Raw dataset:**  
`<ADD GOOGLE DRIVE LINK>`

**Processed dataset:**  
`<ADD GOOGLE DRIVE LINK>`

**Kaggle source:**  
`<ADD KAGGLE DATASET LINK>`

---

## 3. Project Workflow

```text
Raw Dataset
    ↓
Data Loading
    ↓
EDA & Data Quality Analysis
    ↓
Feature Engineering
    ↓
Train / Validation / Test Split
    ↓
 ┌───────────────────────┐
 │                       │
 ▼                       ▼
XGBoost Baseline      TabNet Challenger
 │                       │
 └──────────┬────────────┘
            ▼
      Model Evaluation
            ↓
        ONNX Export
            ↓
     ONNX Runtime Test
            ↓
 Accuracy & Latency Comparison
            ↓
      Streamlit Deployment
```

---

## 4. Exploratory Data Analysis

The original dataset contains:

```text
Rows    : 45,584
Columns : 20
```

After separating the target:

```text
Input features : 19
Target         : Time_taken (min)
```

Important missing values include:

| Feature | Missing values |
|---|---:|
| Delivery_person_Age | 1,854 |
| Delivery_person_Ratings | 1,908 |
| Time_Orderd | 1,731 |
| Weather_conditions | 616 |
| Road_traffic_density | 601 |
| multiple_deliveries | 993 |
| Festival | 228 |
| City | 1,200 |

The target variable has no missing values.

Missing values are handled inside the preprocessing pipeline using:

- **Median imputation** for numerical features
- **Most-frequent imputation** for categorical features

---

## 5. Feature Engineering

Domain-based feature engineering was applied before model training.

### Date features

`Order_Date` is transformed into:

```text
order_day
order_month
order_day_of_week
is_weekend
```

### Time features

`Time_Orderd` is transformed into:

```text
order_hour
order_minute
```

`Time_Order_picked` is transformed into:

```text
pickup_hour
pickup_minute
```

### Pickup delay

The difference between order time and pickup time is calculated as:

```text
pickup_delay
```

### Geographic distance

Restaurant and delivery coordinates are used to calculate approximate Haversine distance:

```text
distance_km
```

After feature engineering:

```text
Original input features  : 19
Engineered input features: 26
```

After preprocessing:

```text
Processed features: 41
```

The same engineered feature representation is used for the XGBoost and TabNet comparison.

---

## 6. Preprocessing

### Numerical features

```text
Median Imputation
        ↓
Standard Scaling
```

### Categorical features

```text
Most-Frequent Imputation
        ↓
One-Hot Encoding
```

Unknown categories during inference are handled with:

```python
handle_unknown="ignore"
```

The fitted preprocessing pipeline is stored together with the XGBoost model, allowing the deployed application to apply the same transformations used during training.

---

## 7. Train / Validation / Test Split

### XGBoost

```text
Training pool : 36,467
Test set      : 9,117
```

### TabNet

The training pool was further divided into:

```text
Training   : 32,820
Validation :  3,647
Test       :  9,117
```

The test set was kept untouched during model selection. TabNet uses the validation set for model selection and early stopping.

---

## 8. Project Structure

Only the main project components are shown below; generated datasets, model binaries, MLflow files, and temporary artifacts are excluded from GitHub.

```text
swiftdrop-delivery-prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_xgboost.py
│   ├── evaluate_xgboost.py
│   ├── evaluate_onnx.py
│   ├── compare_xgboost_onnx.py
│   ├── prepare_pytorch_data.py
│   ├── tabular_dataset.py
│   ├── create_dataloaders.py
│   ├── train_tabnet.py
│   ├── evaluate_tabnet.py
│   ├── export_tabnet_onnx.py
│   └── evaluate_tabnet_onnx.py
│
├── notebooks/
│   └── 01_xgboost_baseline.ipynb
│
└── .gitignore
```

### Key source files

| File | Purpose |
|---|---|
| `src/data_loader.py` | Loads the delivery dataset |
| `src/preprocessing.py` | Feature engineering and sklearn preprocessing |
| `src/train_xgboost.py` | Trains and logs the XGBoost baseline |
| `src/evaluate_xgboost.py` | Evaluates native XGBoost |
| `src/evaluate_onnx.py` | Evaluates XGBoost ONNX with ONNX Runtime |
| `src/compare_xgboost_onnx.py` | Verifies native vs ONNX predictions |
| `src/train_tabnet.py` | Trains the TabNet challenger |
| `src/evaluate_tabnet.py` | Evaluates native TabNet |
| `src/export_tabnet_onnx.py` | Exports TabNet to ONNX |
| `src/evaluate_tabnet_onnx.py` | Evaluates TabNet ONNX |
| `app.py` | Streamlit prediction application |

---

## 9. XGBoost Baseline

XGBoost was used as the primary traditional machine learning baseline for the structured delivery dataset.

Configuration:

```text
n_estimators     = 300
max_depth        = 8
learning_rate    = 0.05
subsample        = 0.8
colsample_bytree = 0.8
objective        = reg:squarederror
```

The complete fitted preprocessing + XGBoost pipeline is stored externally and loaded by the Streamlit application.

---

## 10. TabNet Challenger

TabNet was selected as the neural-network challenger because it is designed for tabular data and uses attention mechanisms to learn relevant feature representations.

Configuration:

```text
n_d            = 32
n_a            = 32
n_steps        = 5
gamma          = 1.5
lambda_sparse  = 1e-4
optimizer      = Adam
learning_rate  = 0.001
max_epochs     = 100
```

Validation-based model selection and early stopping were used during training.

---

## 11. MLflow Experiment Tracking

MLflow is used to track experiments and evaluation results.

Tracked information includes:

- Model parameters
- Model configuration
- Dataset source
- Validation/test metrics
- Model artifacts

Experiment name:

```text
SwiftDrop_Delivery_Prediction
```

---

## 12. Model Evaluation

### XGBoost Native

Evaluated on the complete **9,117-row test set**:

| Metric | Result |
|---|---:|
| MAE | **3.1798 min** |
| RMSE | **4.0053 min** |
| R² | **0.8181** |
| Latency — 1,000 rows | **14.7781 ms** |

### XGBoost ONNX

Evaluated with ONNX Runtime on the complete test set:

| Metric | Result |
|---|---:|
| MAE | **3.2398 min** |
| MSE | **16.4495** |
| RMSE | **4.0558 min** |
| R² | **0.8050** |
| Latency — 1,000 rows | **1.0119 ms** |
| Latency requirement | **50 ms** |
| Status | **PASS** |

### XGBoost Native vs ONNX Verification

For the first 1,000 test rows:

```text
Pipeline vs XGBoost
Maximum difference: 0.0

XGBoost vs ONNX
Maximum difference : 0.00006866
Mean difference    : 0.00001140
```

The same 1,000-row comparison produced:

```text
Native XGBoost MAE : 3.2398
ONNX MAE           : 3.2398
```

This indicates only negligible floating-point differences between the native and ONNX predictions.

### TabNet Native

Evaluated on the complete **9,117-row test set**:

| Metric | Result |
|---|---:|
| MAE | **4.7800 min** |
| MSE | **36.9684** |
| RMSE | **6.0802 min** |
| R² | **0.5809** |
| Latency — 1,000 rows | **24.4266 ms** |
| Latency requirement | **50 ms** |
| Status | **PASS** |

### TabNet ONNX

Evaluated on the 1,000-row ONNX benchmark:

| Metric | Result |
|---|---:|
| MAE | **4.9491 min** |
| MSE | **40.3701** |
| RMSE | **6.3537 min** |
| R² | **0.5215** |
| Latency — 1,000 rows | **11.6322 ms** |
| Latency requirement | **50 ms** |
| Status | **PASS** |

> **Evaluation note:** TabNet ONNX accuracy was measured on the 1,000-row benchmark, while native TabNet accuracy was measured on the complete 9,117-row test set. Therefore, those two TabNet accuracy results should not be treated as a strict apples-to-apples full-test comparison.

---

## 13. Final Model Comparison

| Model | Evaluation Scope | MAE (min) | RMSE (min) | R² | Latency / 1000 |
|---|---|---:|---:|---:|---:|
| XGBoost Native | Full test set | **3.1798** | **4.0053** | **0.8181** | 14.7781 ms |
| XGBoost ONNX | Full test set | 3.2398 | 4.0558 | 0.8050 | **1.0119 ms** |
| TabNet Native | Full test set | 4.7800 | 6.0802 | 0.5809 | 24.4266 ms |
| TabNet ONNX | 1,000-row benchmark | 4.9491 | 6.3537 | 0.5215 | **11.6322 ms** |

All measured ONNX latency results are below the required **50 ms** limit.

---

## 14. Recommendation

The current experiments show that XGBoost achieved the lowest measured delivery-time error on the full test set, with an MAE of **3.1798 minutes**. Its ONNX version also achieved approximately **1.01 ms latency for 1,000 rows**, while the native pipeline measured **14.78 ms**. The native-versus-ONNX verification showed only negligible prediction differences, making the exported XGBoost model suitable for the measured production-style inference requirement.

TabNet also met the 50 ms latency requirement in both native and ONNX Runtime benchmarks and demonstrates a neural tabular alternative. However, the available evaluation results show higher measured error for TabNet than XGBoost. Based on these measured results, the project uses the **XGBoost pipeline for the deployed Streamlit application**, while retaining TabNet as the challenger model used for comparison.

---

## 15. Model Artifacts

Model binaries are **not committed to GitHub**.

The trained XGBoost pipeline is stored externally on Google Drive and downloaded by the deployed Streamlit application at runtime.

### XGBoost model

```text
xgboost_pipeline.pkl
```

### Google Drive model

The deployed application uses this Google Drive file:

```text
1s73I1dYpubS9yg-ZVAdb7UjeIBgO9BRW
```

The Google Drive file is configured for link-based access so the Streamlit application can download it without exposing model credentials in the repository.

---

## 16. Streamlit Deployment

### Live application

**[Open SwiftDrop Delivery Prediction App](https://swiftdrop-delivery-prediction-qnypkftgrupfgjn3tazyrm.streamlit.app/)**

The deployed application:

1. Collects raw delivery/order/location information.
2. Applies the same feature-engineering function used during training.
3. Downloads the XGBoost pipeline from Google Drive when required.
4. Loads the fitted preprocessing + model pipeline.
5. Generates the estimated delivery time.

The model file and Streamlit secrets are kept outside GitHub.

### How the deployment is configured

The repository contains the application code, while the trained model is stored on Google Drive.

A Streamlit Cloud secret is configured as:

```toml
XGBOOST_MODEL_FILE_ID = "1s73I1dYpubS9yg-ZVAdb7UjeIBgO9BRW"
```

The secret is configured through **Streamlit Cloud → App Settings → Secrets** and is not committed to GitHub.

### Test the application

Open:

**https://swiftdrop-delivery-prediction-qnypkftgrupfgjn3tazyrm.streamlit.app/**

Enter sample delivery information and submit the form. The application will display an estimated delivery time in minutes.

---

## 17. Local Installation

Clone the repository:

```bash
git clone https://github.com/Swa990/swiftdrop-delivery-prediction.git
cd swiftdrop-delivery-prediction
```

Create and activate a virtual environment:

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scriptsctivate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

For local inference, the required model artifact must be available locally or downloaded from the configured external storage.

---

## 18. Reproducing the Training Pipeline

### Train XGBoost

```bash
python -m src.train_xgboost
```

### Evaluate XGBoost

```bash
python -m src.evaluate_xgboost
```

### Evaluate XGBoost ONNX

```bash
python -m src.evaluate_onnx
```

### Compare XGBoost Native vs ONNX

```bash
python -m src.compare_xgboost_onnx
```

### Prepare TabNet datasets

```bash
python -m src.prepare_pytorch_data
```

### Train TabNet

```bash
python -m src.train_tabnet
```

### Evaluate TabNet

```bash
python -m src.evaluate_tabnet
```

### Export TabNet to ONNX

```bash
python -m src.export_tabnet_onnx
```

### Evaluate TabNet ONNX

```bash
python -m src.evaluate_tabnet_onnx
```

---

## 19. Latency Requirement

The business requirement is:

```text
Maximum inference latency for 1,000 rows = 50 ms
```

Measured ONNX Runtime results:

```text
XGBoost ONNX : 1.0119 ms → PASS
TabNet ONNX  : 11.6322 ms → PASS
```

Both exported models satisfy the latency requirement.

---

## 20. Technologies Used

### Programming & Data

- Python
- Pandas
- NumPy
- SQL

### Machine Learning

- Scikit-learn
- XGBoost
- PyTorch
- TabNet

### Experiment Tracking

- MLflow

### Model Optimization & Inference

- ONNX
- ONNX Runtime

### Application & Deployment

- Streamlit
- Google Drive
- Streamlit Cloud

### Development

- Git
- GitHub

---

## 21. Reproducibility

The project uses fixed random seeds where applicable, keeps the test set isolated from model selection, stores fitted preprocessing objects with the model pipeline, and tracks experiments using MLflow.

The deployed application reuses the same feature-engineering logic used during model development to reduce training/inference preprocessing differences.

---

## 22. Future Improvements

Potential improvements include:

- Hyperparameter optimization for XGBoost and TabNet
- Additional geospatial features
- More detailed temporal features
- Feature importance and model explainability
- Model monitoring and drift detection
- API-based inference
- Automated CI/CD model deployment
- Production model registry and versioning

---

## 23. Conclusion

SwiftDrop demonstrates an end-to-end machine learning workflow for delivery-time prediction, from data analysis and feature engineering through model training, MLflow tracking, ONNX conversion, latency benchmarking, and cloud deployment.

The measured experiments provide a comparison between a traditional tabular ML approach and a neural tabular approach. XGBoost achieved lower measured prediction error on the full test evaluation, while both ONNX implementations satisfied the 50 ms latency requirement. The final XGBoost pipeline is deployed as an interactive Streamlit application that can be tested directly through the live demo.

### Links

- **GitHub Repository:** https://github.com/Swa990/swiftdrop-delivery-prediction
- **Live Streamlit App:** https://swiftdrop-delivery-prediction-qnypkftgrupfgjn3tazyrm.streamlit.app/
