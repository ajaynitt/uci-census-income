import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

st.title("Adult Income Classification App")

uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

model_choice = st.selectbox(
    "Select Model",
    ["Logistic_Regression",
     "Decision_Tree",
     "KNN",app.py
     "Naive_Bayes",
     "Random_Forest",
     "XGBoost"]
)

if uploaded_file:
    # 1. Define the exact column names the model expects
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    # 2. Read the file with the assigned names
    # skipinitialspace=True handles the spaces after commas in the UCI dataset
    data = pd.read_csv(uploaded_file, names=column_names, skipinitialspace=True)

    # Handle the trailing dot if you are using the 'adult.test' file
    if data['income'].dtype == 'object':
        data['income'] = data['income'].str.replace('.', '', regex=False)

    data = data.dropna()

    for col in data.select_dtypes(include=['object']).columns:
        data[col] = LabelEncoder().fit_transform(data[col])

    X = data.drop("income", axis=1)
    y = data["income"]

    model = joblib.load(f"model/{model_choice}.pkl")
    predictions = model.predict(X)

    st.subheader("Classification Report")
    st.text(classification_report(y, predictions))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y, predictions))
