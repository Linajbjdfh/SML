
import pandas as pd
import pickle # un-pickling stuff from training notebook
from xgboost import XGBRegressor # we use a trained XGBoost model
from sklearn.preprocessing import StandardScaler
import shap # add prediction explainability

import numpy as np
import itertools # we need that to flatten ohe.categories_ into one list for columns
import streamlit as st


st.set_page_config(
    page_title="HR",
    page_icon="💸")

st.title('Predict Attrition')


# use this decorator (--> @st.experimental_singleton) and 0-parameters function to only load and preprocess once
@st.experimental_singleton
def read_objects():
    model_xgb = pickle.load(open('model_xgb.pkl','rb'))
    scaler = pickle.load(open('scaler.pkl','rb'))
    ohe = pickle.load(open('ohe.pkl','rb'))
    shap_values = pickle.load(open('shap_values.pkl','rb'))
    cats = list(itertools.chain(*ohe.categories_))
    return model_xgb, scaler, ohe, cats, shap_values

model_xgb, scaler, ohe, cats, shap_values = read_objects()

with st.expander("What's that app?"):
    st.markdown("""
    This app will help you predict attrition of the employee
    """)

#Creating layout
JobRole = st.selectbox('Select your Job Role', options=ohe.categories_[0])
Gender = st.radio('What is your gender?', options=ohe.categories_[1])
YearsAtCompany = st.number_input('How many years at this company?', min_value=1, max_value=10)
JobSatisfaction = st.number_input('Rate your Job satisfaction?', min_value=1, max_value=4)
NumCompaniesWorked = st.number_input('How many companies you worked at?', min_value=0, max_value=9)

if st.button('Predict! 🚀'):
    # make a DF for categories and transform with one-hot-encoder
    new_values_cat = pd.DataFrame(columns=['Healthcare Representative', 'Human Resources', 'Laboratory Technician',
       'Manager', 'Manufacturing Director', 'Research Director',
       'Research Scientist', 'Sales Executive', 'Sales Representative',
       'Female', 'Male'],dtype='object')

    new_values_cat['Healthcare Representative'] = 0
    new_values_cat['Human Resources'] = 0
    new_values_cat['Laboratory Technician'] = 0
    new_values_cat['Manager'] = 0
    new_values_cat['Manufacturing Director'] = 0
    new_values_cat['Research Director'] = 0
    new_values_cat['Research Scientist'] = 0
    new_values_cat['Sales Executive'] = 0
    new_values_cat['Sales Representative'] = 0
    new_values_cat['Female'] = 0
    new_values_cat['Male'] = 0
    
    new_values_cat[JobRole] = 1
    new_values_cat[Gender] = 1
    

    # make a DF for the numericals and standard scale
    new_df_num = pd.DataFrame({'YearsAtCompany': YearsAtCompany, 
                        'JobSatisfaction':JobSatisfaction, 
                        'NumCompaniesWorked':NumCompaniesWorked 
                        }, index=[0])
    new_values_num = pd.DataFrame(scaler.transform(new_df_num), columns = new_df_num.columns, index=[0])  
    
    #bring all columns together
    line_to_pred = pd.concat([new_values_num, new_values_cat], axis=1)
    
    #run prediction for 1 new observation
    predicted_value = model_xgb.predict(line_to_pred)[0]

    #print out result to user
    st.metric(label="Predicted Attrition", value=f'{predicted_value}')
    
   
