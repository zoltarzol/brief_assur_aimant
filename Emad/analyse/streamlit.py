import streamlit as st 
import pickle
import os
import pandas as pd

def classify_bmi(bmi):
    if bmi < 25:
        return "normal"
    elif bmi < 30:
        return "overweight"
    else:
        return "obese"

# Function to load the model
def load_model():
    model = pickle.load(open('Lasso_Model.pkl', 'rb'))
    st.success('Model loaded successfully')
    return model


model = load_model()



st.title("Abalone Age Prediction")
html_temp = """
<div style="background:#025246 ;padding:10px">
<h2 style="color:white;text-align:center;"> Abalone Age Prediction ML App </h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html = True)


safe_html ="""  
<div style="background-color:#80ff80; padding:10px >
<h2 style="color:white;text-align:center;"> The Abalone is young</h2>
</div>
"""



age = st.number_input("Age (in years)", min_value=1, max_value=100)
height = st.number_input("Height (in cm)", min_value=1)
weight = st.number_input("Weight (in kg)", min_value=1)
sex = st.selectbox("Sex", ["Male", "Female"])
region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])
smoker = st.selectbox("Smoker", ["Yes", "No"])
children = st.number_input("Number of children (if any)", min_value=0, max_value=5)

if st.button('Calculate'):
    bmi = weight / ((height/100) ** 2)
    st.write("Your BMI is: ", bmi)
    bmi_class = classify_bmi(bmi)
    st.write("Your bmi is classified as: ", bmi_class)

    input_data = {'age': age, 'sex': sex, 'region': region, 'smoker': smoker, 'children': children, 'bmi': bmi, 'bmi_class':bmi_class}
    
    input_df = pd.DataFrame(input_data, index=[0])
    input_values = input_df[['age','bmi_class','smoker','children','bmi', 'region', 'sex']].values
    prediction = model.predict(input_values)
    st.write('Predicted Risk of Heart Disease:', prediction)








