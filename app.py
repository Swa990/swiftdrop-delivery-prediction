# import os
# import gdown
# import joblib
# import pandas as pd
# import streamlit as st

# from src.preprocessing import create_features


# # ==================================================
# # 1. Page Configuration
# # ==================================================

# st.set_page_config(
#     page_title="SwiftDrop Delivery Prediction",
#     page_icon="🚴",
#     layout="wide"
# )


# # ==================================================
# # 2. Configuration
# # ==================================================

# MODEL_PATH = "models/xgboost_pipeline.pkl"


# # ==================================================
# # 3. Download Model if Not Available
# # ==================================================

# def download_model():

#     os.makedirs("models", exist_ok=True)

#     if os.path.exists(MODEL_PATH):
#         return

#     # Google Drive file ID stored in Streamlit secrets
#     file_id = st.secrets["XGBOOST_MODEL_FILE_ID"]

#     url = f"https://drive.google.com/uc?id={file_id}"

#     with st.spinner("Downloading trained model..."):

#         gdown.download(
#             url,
#             MODEL_PATH,
#             quiet=False
#         )


# # ==================================================
# # 4. Load Model
# # ==================================================

# @st.cache_resource
# def load_model():

#     download_model()

#     model = joblib.load(
#         MODEL_PATH
#     )

#     return model


# model = load_model()


# # ==================================================
# # 5. Header
# # ==================================================

# st.title("🚴 SwiftDrop Delivery Time Prediction")

# st.write(
#     "Predict the expected delivery time for a food delivery "
#     "order using a trained XGBoost regression pipeline."
# )

# st.divider()


# # ==================================================
# # 6. Delivery Information
# # ==================================================

# st.subheader("Delivery Information")

# col1, col2, col3 = st.columns(3)


# with col1:

#     age = st.number_input(
#         "Delivery Person Age",
#         min_value=18,
#         max_value=60,
#         value=30,
#         step=1
#     )

#     ratings = st.number_input(
#         "Delivery Person Rating",
#         min_value=1.0,
#         max_value=5.0,
#         value=4.5,
#         step=0.1
#     )

#     vehicle_condition = st.number_input(
#         "Vehicle Condition",
#         min_value=0,
#         max_value=5,
#         value=2,
#         step=1
#     )


# with col2:

#     weather = st.selectbox(
#         "Weather Conditions",
#         [
#             "Sunny",
#             "Stormy",
#             "Sandstorms",
#             "Cloudy",
#             "Windy",
#             "Fog"
#         ]
#     )

#     traffic = st.selectbox(
#         "Road Traffic Density",
#         [
#             "Low",
#             "Medium",
#             "High",
#             "Jam"
#         ]
#     )

#     city = st.selectbox(
#         "City",
#         [
#             "Urban",
#             "Metropolitian",
#             "Semi-Urban"
#         ]
#     )


# with col3:

#     order_type = st.selectbox(
#         "Type of Order",
#         [
#             "Snack",
#             "Meal",
#             "Drinks",
#             "Buffet"
#         ]
#     )

#     vehicle_type = st.selectbox(
#         "Type of Vehicle",
#         [
#             "motorcycle",
#             "scooter",
#             "electric_scooter",
#             "bicycle"
#         ]
#     )

#     multiple_deliveries = st.selectbox(
#         "Multiple Deliveries",
#         [
#             0,
#             1,
#             2,
#             3
#         ]
#     )


# # ==================================================
# # 7. Order Information
# # ==================================================

# st.subheader("Order Information")

# col1, col2, col3 = st.columns(3)


# with col1:

#     festival = st.selectbox(
#         "Festival",
#         [
#             "No",
#             "Yes"
#         ]
#     )


# with col2:

#     order_date = st.date_input(
#         "Order Date"
#     )


# with col3:

#     order_time = st.time_input(
#         "Order Time"
#     )


# pickup_time = st.time_input(
#     "Pickup Time"
# )


# # ==================================================
# # 8. Location Information
# # ==================================================

# st.subheader("Location Information")

# col1, col2 = st.columns(2)


# with col1:

#     st.markdown("### Restaurant")

#     restaurant_latitude = st.number_input(
#         "Restaurant Latitude",
#         value=19.076000,
#         format="%.6f"
#     )

#     restaurant_longitude = st.number_input(
#         "Restaurant Longitude",
#         value=72.877700,
#         format="%.6f"
#     )


# with col2:

#     st.markdown("### Delivery Location")

#     delivery_latitude = st.number_input(
#         "Delivery Latitude",
#         value=19.086000,
#         format="%.6f"
#     )

#     delivery_longitude = st.number_input(
#         "Delivery Longitude",
#         value=72.887700,
#         format="%.6f"
#     )


# # ==================================================
# # 9. Prediction
# # ==================================================

# st.divider()

# predict_button = st.button(
#     "Predict Delivery Time",
#     type="primary",
#     use_container_width=True
# )


# if predict_button:

#     # ------------------------------------------------
#     # Create raw input dataframe
#     # ------------------------------------------------

#     input_data = pd.DataFrame({

#         "Delivery_person_Age": [age],

#         "Delivery_person_Ratings": [ratings],

#         "Restaurant_latitude": [
#             restaurant_latitude
#         ],

#         "Restaurant_longitude": [
#             restaurant_longitude
#         ],

#         "Delivery_location_latitude": [
#             delivery_latitude
#         ],

#         "Delivery_location_longitude": [
#             delivery_longitude
#         ],

#         "Order_Date": [
#             order_date.strftime("%d-%m-%Y")
#         ],

#         "Time_Orderd": [
#             order_time.strftime("%H:%M")
#         ],

#         "Time_Order_picked": [
#             pickup_time.strftime("%H:%M")
#         ],

#         "Weather_conditions": [
#             weather
#         ],

#         "Road_traffic_density": [
#             traffic
#         ],

#         "Vehicle_condition": [
#             vehicle_condition
#         ],

#         "Type_of_order": [
#             order_type
#         ],

#         "Type_of_vehicle": [
#             vehicle_type
#         ],

#         "multiple_deliveries": [
#             multiple_deliveries
#         ],

#         "Festival": [
#             festival
#         ],

#         "City": [
#             city
#         ]
#     })


#     # ------------------------------------------------
#     # Feature Engineering
#     # ------------------------------------------------

#     # IMPORTANT:
#     # The XGBoost model was trained on engineered
#     # features, so we must apply the exact same
#     # feature engineering during inference.

#     input_data = create_features(
#         input_data
#     )


#     # ------------------------------------------------
#     # Prediction
#     # ------------------------------------------------

#     prediction = model.predict(
#         input_data
#     )[0]


#     # ------------------------------------------------
#     # Display Result
#     # ------------------------------------------------

#     st.success(
#         f"### Estimated Delivery Time: "
#         f"{prediction:.1f} minutes"
#     )

#     st.info(
#         "The prediction uses the same feature-engineering "
#         "and preprocessing logic used during model training."
#     )


import os
import gdown
import joblib
import pandas as pd
import streamlit as st

from src.preprocessing import create_features


# ============================================================
# 1. Streamlit Configuration
# ============================================================

st.set_page_config(
    page_title="SwiftDrop Delivery Prediction",
    page_icon="🚴",
    layout="wide"
)


# ============================================================
# 2. Paths
# ============================================================

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "xgboost_pipeline.pkl"
)


# ============================================================
# 3. Download Model from Google Drive
# ============================================================

def download_model_from_google_drive():

    os.makedirs(MODEL_DIR, exist_ok=True)

    # If model already exists, don't download again
    if os.path.exists(MODEL_PATH):
        return

    # Read Google Drive file ID from Streamlit Secrets
    file_id = st.secrets["XGBOOST_MODEL_FILE_ID"]

    # Google Drive download URL
    google_drive_url = (
        f"https://drive.google.com/uc?id={file_id}"
    )

    with st.spinner(
        "Downloading trained XGBoost model..."
    ):

        gdown.download(
            google_drive_url,
            MODEL_PATH,
            quiet=False
        )

    # Verify download
    if not os.path.exists(MODEL_PATH):

        st.error(
            "Model download failed. "
            "Please check the Google Drive file ID."
        )

        st.stop()


# ============================================================
# 4. Load Model
# ============================================================

@st.cache_resource
def load_model():

    download_model_from_google_drive()

    model = joblib.load(
        MODEL_PATH
    )

    return model


model = load_model()


# ============================================================
# 5. Application Header
# ============================================================

st.title(
    "🚴 SwiftDrop Delivery Time Prediction"
)

st.write(
    "Predict the expected delivery time for a food "
    "delivery order using a trained XGBoost model."
)

st.divider()


# ============================================================
# 6. Delivery Information
# ============================================================

st.subheader("Delivery Information")

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Delivery Person Age",
        min_value=18,
        max_value=60,
        value=30,
        step=1
    )

    ratings = st.number_input(
        "Delivery Person Rating",
        min_value=1.0,
        max_value=5.0,
        value=4.5,
        step=0.1
    )

    vehicle_condition = st.number_input(
        "Vehicle Condition",
        min_value=0,
        max_value=5,
        value=2,
        step=1
    )


with col2:

    weather = st.selectbox(
        "Weather Conditions",
        [
            "Sunny",
            "Stormy",
            "Sandstorms",
            "Cloudy",
            "Windy",
            "Fog"
        ]
    )

    traffic = st.selectbox(
        "Road Traffic Density",
        [
            "Low",
            "Medium",
            "High",
            "Jam"
        ]
    )

    city = st.selectbox(
        "City",
        [
            "Urban",
            "Metropolitian",
            "Semi-Urban"
        ]
    )


with col3:

    order_type = st.selectbox(
        "Type of Order",
        [
            "Snack",
            "Meal",
            "Drinks",
            "Buffet"
        ]
    )

    vehicle_type = st.selectbox(
        "Type of Vehicle",
        [
            "motorcycle",
            "scooter",
            "electric_scooter",
            "bicycle"
        ]
    )

    multiple_deliveries = st.selectbox(
        "Multiple Deliveries",
        [
            0,
            1,
            2,
            3
        ]
    )


# ============================================================
# 7. Order Information
# ============================================================

st.subheader("Order Information")

col1, col2, col3 = st.columns(3)


with col1:

    festival = st.selectbox(
        "Festival",
        [
            "No",
            "Yes"
        ]
    )


with col2:

    order_date = st.date_input(
        "Order Date"
    )


with col3:

    order_time = st.time_input(
        "Order Time"
    )


pickup_time = st.time_input(
    "Pickup Time"
)


# ============================================================
# 8. Location Information
# ============================================================

st.subheader("Location Information")

col1, col2 = st.columns(2)


with col1:

    st.markdown("### Restaurant Location")

    restaurant_latitude = st.number_input(
        "Restaurant Latitude",
        value=19.076000,
        format="%.6f"
    )

    restaurant_longitude = st.number_input(
        "Restaurant Longitude",
        value=72.877700,
        format="%.6f"
    )


with col2:

    st.markdown("### Delivery Location")

    delivery_latitude = st.number_input(
        "Delivery Latitude",
        value=19.086000,
        format="%.6f"
    )

    delivery_longitude = st.number_input(
        "Delivery Longitude",
        value=72.887700,
        format="%.6f"
    )


# ============================================================
# 9. Prediction
# ============================================================

st.divider()

predict_button = st.button(
    "🚴 Predict Delivery Time",
    type="primary",
    use_container_width=True
)


if predict_button:

    # --------------------------------------------------------
    # Create raw input
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "Delivery_person_Age": [age],

        "Delivery_person_Ratings": [ratings],

        "Restaurant_latitude": [
            restaurant_latitude
        ],

        "Restaurant_longitude": [
            restaurant_longitude
        ],

        "Delivery_location_latitude": [
            delivery_latitude
        ],

        "Delivery_location_longitude": [
            delivery_longitude
        ],

        "Order_Date": [
            order_date.strftime("%d-%m-%Y")
        ],

        "Time_Orderd": [
            order_time.strftime("%H:%M")
        ],

        "Time_Order_picked": [
            pickup_time.strftime("%H:%M")
        ],

        "Weather_conditions": [
            weather
        ],

        "Road_traffic_density": [
            traffic
        ],

        "Vehicle_condition": [
            vehicle_condition
        ],

        "Type_of_order": [
            order_type
        ],

        "Type_of_vehicle": [
            vehicle_type
        ],

        "multiple_deliveries": [
            multiple_deliveries
        ],

        "Festival": [
            festival
        ],

        "City": [
            city
        ]
    })


    # --------------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------------

    input_data = create_features(
        input_data
    )


    # --------------------------------------------------------
    # Model Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        input_data
    )[0]


    # --------------------------------------------------------
    # Display Prediction
    # --------------------------------------------------------

    st.success(
        f"### Estimated Delivery Time: "
        f"{prediction:.1f} minutes"
    )

    st.info(
        "Prediction generated using the trained "
        "SwiftDrop XGBoost pipeline."
    )