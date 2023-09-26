import streamlit as st
import pandas as pd
import numpy as np
import pickle

MODEL_NAME = "/content/age_detector_model (1)"

@st.cache_resource
def load_model(model_name):
    with open(model_name, "rb") as file_name:
        return pickle.load(file_name)


#load the model
tabular_model = load_model(MODEL_NAME)

#creating the web app

#title
st.title("Age Project")

#dashboard
st.subheader("User Dashboard")

num_countries = st.slider("Num_countries",0,9,5)
years_in_school = st.slider("Number of years in the school", 0,20,5)
height = st.slider("Your Height",2.00,7.00,4.00)

input_data = {
    "num_countries": num_countries,
    "years_school": years_in_school,
    "height": height

}

#user data
st.subheader("User Data")
st.dataframe(pd.DataFrame(input_data, index = [0]))

#predicting
labels = ["Adult", "child"]
index = tabular_model.predict(np.array([[num_countries, years_in_school, height]]))
who_am_i = labels[index[0]]
print(who_am_i)
st.subheader("Predictions")
if who_am_i == 'child':
    st.write("you are a **Child**")
else:
    st.write("You are an **Adult**")
