# app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd


# Load the model and preprocessors

model = pickle.load(open("XGBoost_best_model.pkl", "rb"))
encoders = pickle.load(open("encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="ShipmentSure - On-Time Delivery Prediction", page_icon="🚚")


# App Title and Description

st.title(" ShipmentSure: Predicting On-Time Delivery")
st.write("### Enter shipment details below to predict whether it will be delivered on time or delayed.")


# Input Form

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        Warehouse_block = st.selectbox("Warehouse Block", ["A", "B", "C", "D", "E"])
        Mode_of_Shipment = st.selectbox("Mode of Shipment", ["Ship", "Flight", "Road"])
        Customer_care_calls = st.number_input("Customer Care Calls", min_value=1, max_value=10, step=1)
        Customer_rating = st.slider("Customer Rating", 1, 5)
        Cost_of_the_Product = st.number_input("Cost of Product (₹)", min_value=50, max_value=10000)

    with col2:
        Prior_purchases = st.number_input("Prior Purchases", min_value=0, max_value=20)
        Product_importance = st.selectbox("Product Importance", ["low", "medium", "high"])
        Gender = st.selectbox("Customer Gender", ["M", "F"])
        Discount_offered = st.number_input("Discount Offered (%)", min_value=0, max_value=100)
        Weight_in_gms = st.number_input("Weight of Product (grams)", min_value=100, max_value=20000)

    submit_button = st.form_submit_button("Predict Delivery Status")


# On Submit — Prediction

if submit_button:
    # Step 1: Create a DataFrame from user input
    input_data = pd.DataFrame({
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

    # Step 2: Encode categorical features (same as during training)
    input_data['Product_importance'] = encoders['Product_importance'].transform(input_data['Product_importance'])
    input_data['Gender'] = encoders['Gender'].transform(input_data['Gender'])

    # Step 3: One-hot encode nominal columns (Warehouse_block & Mode_of_Shipment)
    input_data = pd.get_dummies(input_data, columns=['Warehouse_block', 'Mode_of_Shipment'], drop_first=True)

    # Step 4: Align with training features (handle missing dummy columns)
    expected_features = model.get_booster().feature_names
    for col in expected_features:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[expected_features]  # ensure correct column order

    # Step 5: Scale numerical features
    num_cols = ['Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product',
                'Prior_purchases', 'Discount_offered', 'Weight_in_gms']
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # Step 6: Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Step 7: Display results
    st.subheader(" Prediction Results")
    if prediction == 1:
        st.success(" The shipment is likely to be **delivered on time**!")
    else:
        st.error(" The shipment is likely to be **delayed**.")

    st.write(f"**Prediction Value:** {int(prediction)} (0 = Delayed, 1 = On Time)")
    st.write(f"**Probability of On-Time Delivery:** {probability:.2f}")
