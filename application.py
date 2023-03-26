import streamlit as st
import streamlit.components.v1 as components

from PIL import Image
import time

import pandas as pd 
import numpy as np

import pickle

#THE ENC IS USED TO KEEP THE CATEGORICAL ATTRIBUTES FROM PRE PROCESSING

enc = pickle.load(open("final_model_credit_score_enc", "rb"))

#LE IS THE LABEL ENCODER THAT TAKES THE NON NUMERICAL ATTRIBUTES AND CONVERTS THEM TO NUMERICAL ATTRIBUTES

le = pickle.load(open("final_model_credit_score_le", "rb"))

#MODEL IS THE TRAINED MODEL THAT IS THEN USED TO RETURN THE PREDICTION

model = pickle.load(open("final_model_credit_score_pred", "rb"))





st.markdown("<h2 style='text-align:center; color:floralWhite;'> CREDIT SCORE USING MACHINE LEARNING", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,8,1])

try:
    img1 = Image.open("image3.jpg")
    

    with col2:
        st.image(img1, caption = "Credit Risk Analysis")
        st.markdown('[PROJECT TARS](https://github.com/sangam-kumar)')


except:
    components.html('''
    <script>
        alert("Image Not Loading")
    </script>
    ''')
    st.text("Image Not Loading")

else:
    pass

finally:
    pass

st.sidebar.header("User input parameter")

def user_input_data():
    Credit_Mix = st.sidebar.selectbox('Credit_Mix:', ['Standard', 'Bad', 'Good'])
    Interest_Rate = st.sidebar.slider('Interest_Rate', 1, 34, 14, 1)
    Outstanding_Debt = st.sidebar.slider('Outstanding_Debt', 0, 5000, 1426, 1)
    Delay_from_due_date = st.sidebar.slider('Delay_from_due_date', 0, 62, 21, 1)
    Total_EMI_per_month = st.sidebar.slider('Total_EMI_per_month', 0, 2000, 107, 1)
    Changed_Credit_Limit = st.sidebar.slider('Changed_Credit_Limit', 0, 30, 10, 1)
    Monthly_Inhand_Salary = st.sidebar.slider('Monthly_Inhand_Salary', 303, 15000, 4197, 1)
    Annual_Income = st.sidebar.slider('Annual_Income', 7000, 180000, 50505, 1)
    
    html_temp = """
    <div style="background-color:teal;padding:1.5px">
    <h1 style="color:black;text-align:center;">Single Customer </h1>
    </div><br>"""
    st.sidebar.markdown(html_temp,unsafe_allow_html=True)
    
    data = { 
        'Credit_Mix': Credit_Mix,
        'Interest_Rate': Interest_Rate,
        'Outstanding_Debt': Outstanding_Debt,
        'Delay_from_due_date': Delay_from_due_date,
        'Total_EMI_per_month': Total_EMI_per_month,
        'Changed_Credit_Limit': Changed_Credit_Limit,
        'Monthly_Inhand_Salary': Monthly_Inhand_Salary,
        'Annual_Income': Annual_Income,
    }
    input_data = pd.DataFrame(data, index=[0])  
    
    return input_data


#show input
col1, col2 = st.columns([4, 6])

df = user_input_data() 
with col1:
    if st.checkbox('Show User Inputs:', value=True):
        st.write(df.astype(str).T.rename(columns={0:'input_data'}))

with col2:
    for i in range(2): 
        st.markdown('#')
        cat = ['Credit_Mix']
        df[cat] = enc.transform(df[cat]) 
        prediction = model.predict(df)
        prediction = le.inverse_transform(prediction)[0]

        time.sleep(3)  # wait for 2 seconds
        
        st.success(f'Credit score probability is:&emsp;{prediction}')