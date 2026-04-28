import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

#load data
df=pd.read_csv("train(1).csv")

#feature enineering
df['TotalIncome']=df['ApplicantIncome']+df['CoapplicantIncome']
df['Income_Loan_Ratio']=df['TotalIncome']/df['LoanAmount']

#prepare data

X=df[['Credit_History','ApplicantIncome','LoanAmount_log','TotalIncome_log','Income_Loan_Ratio']]
y=df['Loan_Status'].map({'Y':1,'N':0})

#handel missing values
X= X.fillna(X.mean())

#Sclaing
scaler= StandardScaler()
X_scaled=scaler.fit_transform(X)

# Train model
model = RandomForestClassifier()
model.fit(X_scaled, y)

# UI
st.title("🏦 Loan Approval Prediction")

credit = st.selectbox("Credit History (1 = Good, 0 = Bad)", [1,0])
income = st.number_input("Applicant Income")
loan = st.number_input("Loan Amount")
co_income = st.number_input("Coapplicant Income")

total_income = income + co_income
ratio = total_income / loan if loan != 0 else 0

input_data = np.array([[credit, income, loan, total_income, ratio]])
input_scaled = scaler.transform(input_data)

if st.button("Predict"):
    result = model.predict(input_scaled)
    
    if result[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")




