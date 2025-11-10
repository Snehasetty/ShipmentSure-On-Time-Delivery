import pickle
import streamlit as st
import numpy as np
import pandas as pd

# -------------------------------
# Load Model and Preprocessors
# -------------------------------
model = pickle.load(open("XGBoost_best_model.pkl", "rb"))
encoders = pickle.load(open("encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="ShipmentSure - On-Time Delivery Prediction", page_icon="🚚")
st.title("🚚 ShipmentSure: On-Time Delivery Prediction")
st.write("### Enter shipment details below to predict whether the delivery will be **on time** or **delayed.**")

# -------------------------------
# Input Section
# -------------------------------
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        Warehouse_block = st.selectbox("Warehouse Block", ["A", "B", "C", "D", "F"])
        Mode_of_Shipment = st.selectbox("Mode of Shipment", ["Ship", "Flight", "Road"])
        Customer_care_calls = st.number_input("Customer Care Calls", min_value=2, max_value=7, value=4)
        Customer_rating = st.slider("Customer Rating", min_value=1, max_value=5, value=3)
        Cost_of_the_Product = st.number_input("Cost of Product (₹)", min_value=96, max_value=800, value=214, step=1)

    with col2:
        Prior_purchases = st.number_input("Prior Purchases", min_value=2, max_value=10, value=3)
        Product_importance = st.selectbox("Product Importance", ["low", "medium", "high"])
        Gender = st.selectbox("Customer Gender", ["M", "F"])
        Discount_offered = st.number_input("Discount Offered (%)", min_value=1, max_value=65, value=10)
        Weight_in_gms = st.number_input("Weight of Product (grams)", min_value=1001, max_value=7846, value=4149, step=100)

    submit = st.form_submit_button("Predict Delivery Status")

# -------------------------------
# Prediction Logic
# -------------------------------
if submit:
    # Step 1: Create DataFrame with RAW values
    df = pd.DataFrame({
        'Warehouse_block': [Warehouse_block],
        'Mode_of_Shipment': [Mode_of_Shipment],
        'Customer_care_calls': [Customer_care_calls],
        'Customer_rating': [Customer_rating],
        'Cost_of_the_Product': [Cost_of_the_Product],
        'Prior_purchases': [Prior_purchases],
        'Product_importance': [Product_importance],
        'Gender': [Gender],
        'Discount_offered': [Discount_offered],
        'Weight_in_gms': [Weight_in_gms]
    })

    # Step 2: Label Encode
    df['Product_importance'] = encoders['Product_importance'].transform(df['Product_importance'])
    df['Gender'] = encoders['Gender'].transform(df['Gender'])

    # Step 3: One-Hot Encode Nominal Columns
    df = pd.get_dummies(df, columns=['Warehouse_block', 'Mode_of_Shipment'], drop_first=True)

    # Step 4: Feature Engineering (raw values, same as training)
    df['Cost_to_Weight_ratio'] = df['Cost_of_the_Product'] / df['Weight_in_gms']
    df['Cost*Weight'] = df['Cost_of_the_Product'] * df['Weight_in_gms']
    df['Discount_Ratio'] = df['Discount_offered'] / df['Cost_of_the_Product']

    # Step 5: Handle NaN or Inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Step 6: Scale ONLY base numerical columns (as per training)
    base_num_cols = ['Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product',
                     'Prior_purchases', 'Discount_offered', 'Weight_in_gms']
    df[base_num_cols] = scaler.transform(df[base_num_cols])

    # Step 7: Align with model features
    expected = model.get_booster().feature_names
    for col in expected:
        if col not in df.columns:
            df[col] = 0
    df = df[expected]

    # Step 8: Predict
    DECISION_THRESHOLD = 0.50
    prob = model.predict_proba(df)[0][1]
    if prob >= DECISION_THRESHOLD:
        prediction = 1  # Reached on Time
    else:
        prediction = 0  # Delayed

    # Step 9: Output
    if prediction is not None:
        if prediction == 1:
            st.success(f"**Prediction: SHIPMENT WILL REACH ON TIME!**")
            result_text = "The model predicts the shipment will be delivered successfully on time. This is a low-risk shipment."
        else:
            st.warning(f"**Prediction: SHIPMENT WILL BE DELAYED.**")
            result_text = "The model suggests this shipment is likely to be delayed. **Immediate action is recommended.**"      

        st.info(f"Confidence Score (Probability of On-Time): **{prob*100:.3f}%**")
        st.write(result_text)

    