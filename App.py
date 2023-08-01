import streamlit as st
from pickle import load
import pandas as pd 
import numpy as np
import matplotlib.image as mp
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

st.title("***SALARY IDENTIFICATION***")
st.sidebar.header('User Input Parameters')

# Sidebar Input
YearsOfExperience = st.sidebar.number_input('Enter Your Experience ', 0, 20)
submit = st.sidebar.button('Submit')


# Step 1: Load Data
dataset = pd.read_csv('./Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,1].values

# Step 2: Split data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Step 3: Fit Simple Linear Regression to Training Data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Step 4: Make Prediction
y_pred = regressor.predict(X_test)

# Step 7 - Make new prediction
def result():
    prediction = regressor.predict([[YearsOfExperience]])
    return prediction
        
results = result()
        
if submit is True:
    st.write(results)
