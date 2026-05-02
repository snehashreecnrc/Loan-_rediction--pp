import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

#Page config
st.set_page_config(page_title="Loan Prediction" , layout="centered")


#Title
st.title("🏦Loan Approval Prediction")
st.write("This app predicts loan approval based on financial and credit details.")

#load data
df=pd.read_csv("train (1).csv")

#feature enineering
df['TotalIncome']=df['ApplicantIncome']+df['CoapplicantIncome']
df['Income_Loan_Ratio']=df['TotalIncome']/df['LoanAmount']


#prepare data

X=df[['Credit_History','ApplicantIncome','LoanAmount','TotalIncome','Income_Loan_Ratio']]
y=df['Loan_Status'].map({'Y':1,'N':0})

#handel missing values
X= X.fillna(X.mean())

#Sclaing
scaler= StandardScaler()
X_scaled=scaler.fit_transform(X)

# Train model
model = RandomForestClassifier()
model.fit(X_scaled, y)

#-------------------Feature importance ------------------
importance= model.feature_importances_
features=X.columns
imp_df= pd.DataFrame({
    'Feature':features,
    'Importance': importance 
}).sort_values(by='Importance', ascending=False)

st.subheader("📊 Feature Importance")
st.bar_chart(imp_df.set_index('Feature'))



#------------------User Input-------------------
st.subheader("Enter Applicant Details")

credit = st.selectbox("Credit History (1 = Good, 0 = Bad)", [1,0])
income = st.number_input("Applicant Income")
loan = st.number_input("Loan Amount")
co_income = st.number_input("Coapplicant Income")
obligation = st.number_input("Existing Monthly Obligations (EMI)")

total_income = income + co_income
ratio = total_income / loan if loan != 0 else 0

foir=(obligation+loan)/total_income if total_income != 0 else 0

input_data = np.array([[credit, income, loan, total_income, ratio]])
input_scaled = scaler.transform(input_data)

#---------------------Prediction--------------
if st.button("Predict"):

    if foir > 0.5:
        st.warning("❌ High FOIR(>50%)- Risky Applicant")
    #Business rule
    if credit==0:
        st.error("❌ High Risk: Poor Credit History")
    proba = model.predict_proba(input_scaled)
    
    if proba[0][1]>0.6:
        st.success( f"✅ Loan Approved ({proba[0][1]*100:.2f}% confidence)")
    else:
        st.error(f"❌ Loan Rejected({proba[0][1]*100:.2f}% confidence)")
     






