import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

st.set_page_config('Power Consumption System')
st.title('Power Consumption Forecast')
st.write('Enter the following inputs to predict next 10-min power consumption')


model=load_model('power_lstm.h5',compile=False)
close_idx=pickle.load(open('closeidx.pkl','rb'))
features=pickle.load(open('features.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))
WINDOW=144

def power_safety_label(power_value):
    if power_value < 20000:
        return "Low Consumption"
    elif power_value < 35000:
        return "Moderate Consumption"
    elif power_value < 50000:
        return "High Consumption"
    else:
        return "Very High Consumption - Consider reducing Usage"
    

st.subheader('Enter Current Conditions')

col1,col2=st.columns(2)

with col1:
    power_input=st.number_input("Current Power Zone 1",0.0,100000.0,30000.0,500.0)
    temp_input=st.number_input('Current Temperature',-10.0,50.0,20.0)
    humid_input=st.number_input('Current Humidity Status',0.0,100.0,60.0)

with col2:
    wind_input=st.number_input('Current Wind Speed',0.0,6.483000,0.1,0.001)
    general_input=st.number_input('general Diffuse input',0.0,1200.0,0.05,0.001)
    diffuse_input=st.number_input('Diffuse input',0.0,950.0,0.05,0.001)

hour_input=st.slider('Enter The Hour',0,23,12)
day_input=st.selectbox("Day of Week",options=list(range(7)),format_func=lambda x:['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
month_input=st.selectbox('MOnth',list(range(1,13)),format_func=lambda x:['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
is_weekend= 1 if day_input >= 5 else 0

if st.button('Predict Next 10-min Power Consumption'):
    user_map={
    'PowerConsumption_Zone1':power_input,
    'Temperature':temp_input,
    'Humidity':humid_input,
    'WindSpeed':wind_input,
    'GeneralDiffuseFlows':general_input,
    'DiffuseFlows':diffuse_input,
    'hour':hour_input,
    'dayofweek':day_input,
    'month':month_input,
    'is_weekend':is_weekend,
    'Zone1_MA144':power_input,
    'Zone1_MA1008':power_input
    }

    row=np.array([user_map[f] for f in features])
    sequence=np.tile(row,(WINDOW,1))
    scaled_sequence=scaler.transform(sequence)
    X_input=np.expand_dims(scaled_sequence,axis=0)

    pred_scaled=model.predict(X_input)

    n_features=len(features)
    dummy=np.zeros((1,n_features))
    dummy[:,close_idx]=pred_scaled
    pred_power=scaler.inverse_transform(dummy)[:,close_idx][0]

    st.metric(label="Predicted Next-Interval Power Zone(1)" ,value=f"{pred_power:,.0f} kW")

    status=power_safety_label(pred_power)

    if "Low" in status:
        st.succes(f"{status}")
    elif "Moderate" in status:
        st.info(f"{status}")
    else:
        st.warning(f"{status}")
else:
    st.info("fill in the Above values and Click Predict..")

















