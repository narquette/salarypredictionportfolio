import streamlit as st 
import pandas as pd
import numpy as np
import pickle
import os
from config import company_ids, job_types, degrees, majors, industries

# add title
st.title('Enter the job information below')

# get required data for the prediction model
milesFromMetropolis = st.slider('Miles From Metropolis', 0, 100, 20)
yearsExperience = st.slider('Years Experience', 0, 30, 5)
company = st.selectbox("Select Company", list(zip(*company_ids))[1])
jobtype = st.selectbox("Select JobType", list(zip(*job_types))[1])
degree = st.selectbox("Select Degrees", list(zip(*degrees))[1])
major = st.selectbox("Select Major", list(zip(*majors))[1])
industry = st.selectbox("Select Industry", list(zip(*industries))[1])

# load model
salary_model = pickle.load(open('LGboost.sav', 'rb'))

# build out dataframe to be used for prediction
salary_data = {'companyId': company,
     'jobType': jobtype,
     'degree': degree,
     'major': major,
     'industry': industry,
     'yearsExperience': yearsExperience,
     'milesFromMetropolis': milesFromMetropolis,
     'IsAdvancedDegree': 0,
     'VicePresidentOrAbove': 0}

salary_df = pd.DataFrame(salary_data, index=[0])

prediction = salary_model.predict(salary_df)

'The predicted salary is ', round(prediction[0],2)*1000