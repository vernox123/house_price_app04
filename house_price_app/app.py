import streamlit as st
import joblib
import pandas as pd
import os
st.write("Files:", os.listdir())
st.write("Models folder:", os.listdir("models") if os.path.exists("models") else "No models folder")


st.set_page_config(page_title='House Price Predictor', layout='centered')
st.title('🏡 House Price Predictor (California)')

MedInc = st.number_input('Median Income', value=3.0)
HouseAge = st.number_input('House Age', value=20.0)
AveRooms = st.number_input('Average Rooms', value=5.0)
AveBedrms = st.number_input('Average Bedrooms', value=1.0)
Population = st.number_input('Population', value=1000.0)
AveOccup = st.number_input('Average Occupants', value=3.0)
Latitude = st.number_input('Latitude', value=34.0)
Longitude = st.number_input('Longitude', value=-118.0)

input_df = pd.DataFrame([{
    'MedInc': MedInc,
    'HouseAge': HouseAge,
    'AveRooms': AveRooms,
    'AveBedrms': AveBedrms,
    'Population': Population,
    'AveOccup': AveOccup,
    'Latitude': Latitude,
    'Longitude': Longitude
}])

if st.button('Predict'):
    model = joblib.load('./models/best_model.pkl')
    pred = model.predict(input_df)[0]
    st.success(f'Predicted price: {pred:,.3f}')
