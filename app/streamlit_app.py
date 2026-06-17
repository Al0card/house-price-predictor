import streamlit as st
import json
import requests



url = "http://127.0.0.1:8000/predict"

st.title("House Price Predictor")

text = st.text_area("House Features (JSON)")

clicked = st.button("Predict")



if clicked:
    try:
        parsed_data = json.loads(text)
   
        response = requests.post(url, json={"input_data": parsed_data})
        json_response = response.json()
        st.success(f"Predicted Price: ${json_response['predicted_price']:,}")
    except json.JSONDecodeError as e:
       st.error("Invalid JSON")