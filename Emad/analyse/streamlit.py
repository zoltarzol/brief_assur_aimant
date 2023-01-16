import streamlit as st
import pandas as pd
import numpy as np
import pickle
from my_functions import *
from PIL import Image

st.set_page_config(page_icon=":bar_chart:",
                layout="wide")

pickle_in = open('Lasso_Model.pkl', 'rb') 
modelLasso = pickle.load(pickle_in)


def make_grid(cols,rows):
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid


mygrid0 = make_grid(1,3)
image = Image.open("Simplon.png")
new_image = image.resize((100, 100))
mygrid0[0][0].image(new_image)

title = "Insurance prediction"
original_title = '<p style="font-size: 50px; text-align:center;">{} </p>'.format(title)
mygrid0[0][1].markdown(original_title, unsafe_allow_html=True)




columns_model = ['age', 'sex','bmi','children', 'smoker', 'region','bmi_class']
caracteristique_individue = [0 for _ in range(7)]


mygrid1 = make_grid(1,7)

with mygrid1[0][0]:
    age = st.number_input(label="Age :",min_value=18,step=1, value=18, key="age")
    caracteristique_individue[0] = age

with mygrid1[0][1]:
    sexe = st.radio(
        "Sex :",
        ('male','female' ))
    if sexe == 'male':
        caracteristique_individue[1] = 'male'
    elif sexe == 'female':
        caracteristique_individue[1] = 'female'


with mygrid1[0][2]:
    children  = st.number_input(label="N Children :",value=0,min_value=0,max_value=5, key="children")
    caracteristique_individue[3] = children

with mygrid1[0][3]:
    smoker = st.radio(
        "Smoker ? :",
        ('yes','no' ))
    if smoker == 'yes':
        caracteristique_individue[4] = 'yes'
    elif smoker == 'no':
        caracteristique_individue[4] = 'no'
    st.write("")
    st.write("")
    st.write("")
    button = st.button("Prediction")

with mygrid1[0][4]:
    liste_region= ['northeast', 'southeast', 'southwest', 'northwest']
    region = st.selectbox("Region: ", 
                        liste_region)

    caracteristique_individue[5] = region


with mygrid1[0][5]:
    height = st.number_input("Height in cm? :", min_value=1, value=180)

with mygrid1[0][6]:
    weight = st.number_input("Weight in kg? :", min_value=1, value=70)
    bmi = (weight / ((height/100) ** 2))
    caracteristique_individue[2] = bmi

    if bmi < 25:
        bmi_class = "normal"
    elif bmi < 30:
        bmi_class = "overweight"
    else:
        bmi_class =  "obese"
    caracteristique_individue[6] = bmi_class

mygrid2 = make_grid(1,3)

with mygrid2[0][1]:
    st.write("")
    st.write("")
    st.write("")
    
    if button:
        predic_lasso = int(modelLasso.predict(pd.DataFrame(np.array(caracteristique_individue).reshape(1, -1),columns=columns_model)))
        st.markdown(f" ### Lasso Charges prediction : {predic_lasso} $")


       





    