import streamlit as st
import pickle
import numpy as np

with open("marksModel.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Student Grade Prediction System")
st.write("Predict **Test Grade, Status, and Result** using a pre-trained Linear Regression model")

hours = st.number_input(
    "Enter Hours Studied",
    min_value=0.0,
    max_value=15.0,
    step=0.5
)

# Prediction
if st.button("Predict"):
    prediction = model.predict(np.array([[hours]]))[0]
    status = "Pass" if prediction >= 70 else "Fail"

    if prediction >= 90:
        result = "A"
    elif prediction >= 80:
        result = "B"
    elif prediction >= 70:
        result = "C"
    else:
        result = "D"

    st.success(f"Predicted Test Grade: {prediction:.2f}")
    st.info(f"Status: {status}")
    st.warning(f"Result: {result}")
