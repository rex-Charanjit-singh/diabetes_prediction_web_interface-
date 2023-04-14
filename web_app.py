# -*- coding: utf-8 -*-
"""
Edited on 14/04/2023

by CHARANJIT SINGH 
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved models

diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))


st.title('Diabetes Prediction using ML')
# getting the input data from the user
col1, col2, col3 = st.columns(3)
    
with col1:
    Glucose = st.text_input('Glucose Level')
        
with col2:
    BloodPressure = st.text_input('Blood Pressure value')

with col3:
    Insulin = st.text_input('Insulin Level')

with col1:
    BMI = st.text_input('BMI value')
    
with col2:
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
with col3:
    Age = st.text_input('Age of the Person')    
    
    
    
# code for Prediction
diab_diagnosis = ''
    
# creating a button for Prediction
    
if st.button('Diabetes Test Result'):
    
    
    input_data = (Glucose, BMI, DiabetesPedigreeFunction, Age)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data,dtype=float)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    diab_prediction = diabetes_model.predict(input_data_reshaped)
    #diab_prediction = diabetes_model.predict([[Glucose, BMI, DiabetesPedigreeFunction, Age]])
        
    if (diab_prediction[0] == 1):
        diab_diagnosis = 'The person is diabetic'
    else:
        diab_diagnosis = 'The person is not diabetic'
        
st.success(diab_diagnosis)
