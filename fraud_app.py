import streamlit as st
import pandas as pd
import joblib
import base64

# Paths
MODEL_PATH = "Fraud_detection_project/models/best_model.pkl"
BACKGROUND_IMAGE = "Fraud_detection_project/data/bg_card.png"  # place your image here

st.set_page_config(
    page_title="💳 Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to set background
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background: url(data:image/png;base64,{encoded});
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .main {{
        background-color: rgba(0, 0, 0, 0.6);
        border-radius: 15px;
        padding: 20px;
    }}
    .result-box {{
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }}
    .fraud {{
        background: linear-gradient(45deg, #ff4c4c, #b30000);
        color: white;
    }}
    .legit {{
        background: linear-gradient(45deg, #4caf50, #006400);
        color: white;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Apply background
set_background(BACKGROUND_IMAGE)

# Load model
model = joblib.load(MODEL_PATH)

# Title
st.markdown("<h1 style='text-align: center; color:white;'>💳 Credit Card Fraud Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# File uploader
uploaded_file = st.file_uploader("📂 Upload a CSV file with transactions", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Predictions
    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    df["Fraud_Prediction"] = preds
    df["Fraud_Probability"] = probs

    # Show results
    st.subheader("🔎 Prediction Results")
    for i, row in df.iterrows():
        if row["Fraud_Prediction"] == 1:
            st.markdown(
                f"<div class='result-box fraud'>❌ Transaction {i} is FRAUD with probability {row['Fraud_Probability']:.2%}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-box legit'>✅ Transaction {i} is LEGIT with probability {1-row['Fraud_Probability']:.2%}</div>",
                unsafe_allow_html=True
            )

    # Show styled dataframe
    st.subheader("📊 Transactions with Predictions")
    st.dataframe(df.style.highlight_max(color="red", subset=["Fraud_Probability"]))
