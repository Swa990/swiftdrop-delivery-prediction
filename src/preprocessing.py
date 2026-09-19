# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder


# NUMERIC_COLUMNS = [
#     "Delivery_person_Age",
#     "Delivery_person_Ratings",
#     "Restaurant_latitude",
#     "Restaurant_longitude",
#     "Delivery_location_latitude",
#     "Delivery_location_longitude",
#     "Vehicle_condition",
#     "multiple_deliveries"
# ]


# CATEGORICAL_COLUMNS = [
#     "Weather_conditions",
#     "Road_traffic_density",
#     "Type_of_order",
#     "Type_of_vehicle",
#     "Festival",
#     "City"
# ]


# def create_preprocessor(X):

#     numeric_columns = [
#         column
#         for column in NUMERIC_COLUMNS
#         if column in X.columns
#     ]

#     categorical_columns = [
#         column
#         for column in CATEGORICAL_COLUMNS
#         if column in X.columns
#     ]

#     numeric_pipeline = Pipeline([
#         ("imputer", SimpleImputer(strategy="median")),
#         ("scaler", StandardScaler())
#     ])

#     categorical_pipeline = Pipeline([
#         ("imputer", SimpleImputer(strategy="most_frequent")),
#         ("encoder", OneHotEncoder(handle_unknown="ignore"))
#     ])

#     preprocessor = ColumnTransformer([
#         ("numeric", numeric_pipeline, numeric_columns),
#         ("categorical", categorical_pipeline, categorical_columns)
#     ])

#     return preprocessor


# using features engieering task



import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# --------------------------------------------------
# 1. Feature Engineering
# --------------------------------------------------

def create_features(df):
    """
    Create meaningful date, time, distance, and delivery-related
    features from the raw SwiftDrop dataset.
    """

    df = df.copy()

    # ----------------------------------------------
    # Order Date Features
    # ----------------------------------------------

    if "Order_Date" in df.columns:

        order_date = pd.to_datetime(
            df["Order_Date"],
            errors="coerce",
            dayfirst=True
        )

        df["order_day"] = order_date.dt.day
        df["order_month"] = order_date.dt.month
        df["order_day_of_week"] = order_date.dt.dayofweek
        df["is_weekend"] = (
            order_date.dt.dayofweek >= 5
        ).astype(float)

    # ----------------------------------------------
    # Order Time Features
    # ----------------------------------------------

    if "Time_Orderd" in df.columns:

        order_time = pd.to_datetime(
            df["Time_Orderd"],
            format="%H:%M",
            errors="coerce"
        )

        df["order_hour"] = order_time.dt.hour
        df["order_minute"] = order_time.dt.minute

    # ----------------------------------------------
    # Pickup Time Features
    # ----------------------------------------------

    if "Time_Order_picked" in df.columns:

        pickup_time = pd.to_datetime(
            df["Time_Order_picked"],
            format="%H:%M",
            errors="coerce"
        )

        df["pickup_hour"] = pickup_time.dt.hour
        df["pickup_minute"] = pickup_time.dt.minute

    # ----------------------------------------------
    # Pickup Delay
    # ----------------------------------------------

    if (
        "Time_Orderd" in df.columns
        and "Time_Order_picked" in df.columns
    ):

        order_minutes = (
            df["order_hour"] * 60
            + df["order_minute"]
        )

        pickup_minutes = (
            df["pickup_hour"] * 60
            + df["pickup_minute"]
        )

        pickup_delay = pickup_minutes - order_minutes

        # Handle midnight crossing
        pickup_delay = np.where(
            pickup_delay < 0,
            pickup_delay + 24 * 60,
            pickup_delay
        )

        df["pickup_delay"] = pickup_delay

    # ----------------------------------------------
    # Delivery Distance
    # ----------------------------------------------

    required_coordinates = [
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude"
    ]

    if all(column in df.columns for column in required_coordinates):

        lat1 = np.radians(df["Restaurant_latitude"])
        lon1 = np.radians(df["Restaurant_longitude"])

        lat2 = np.radians(df["Delivery_location_latitude"])
        lon2 = np.radians(df["Delivery_location_longitude"])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1)
            * np.cos(lat2)
            * np.sin(dlon / 2) ** 2
        )

        c = 2 * np.arcsin(np.sqrt(a))

        # Earth radius in kilometers
        earth_radius = 6371

        df["distance_km"] = earth_radius * c

    # ----------------------------------------------
    # Remove raw date/time columns
    # ----------------------------------------------

    df = df.drop(
        columns=[
            "Order_Date",
            "Time_Orderd",
            "Time_Order_picked"
        ],
        errors="ignore"
    )

    return df


# --------------------------------------------------
# 2. Feature Lists
# --------------------------------------------------

NUMERIC_COLUMNS = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",

    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",

    "Vehicle_condition",
    "multiple_deliveries",

    # Engineered date features
    "order_day",
    "order_month",
    "order_day_of_week",
    "is_weekend",

    # Engineered time features
    "order_hour",
    "order_minute",
    "pickup_hour",
    "pickup_minute",
    "pickup_delay",

    # Engineered geographical feature
    "distance_km"
]


CATEGORICAL_COLUMNS = [
    "Weather_conditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City"
]


# --------------------------------------------------
# 3. Preprocessing Pipeline
# --------------------------------------------------

def create_preprocessor(X):

    numeric_columns = [
        column
        for column in NUMERIC_COLUMNS
        if column in X.columns
    ]

    categorical_columns = [
        column
        for column in CATEGORICAL_COLUMNS
        if column in X.columns
    ]

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numeric_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ])

    return preprocessor