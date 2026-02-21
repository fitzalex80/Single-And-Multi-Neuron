# frontend/streamlit_app.py

import streamlit as st
import requests

st.title("Neural Network Demo (Classical Implementation)")

st.sidebar.header("Select Model")
model = st.sidebar.radio("Choose:", ["Single Neuron", "1 Layer (3 Neurons)"])

st.header("Enter Inputs")

x1 = st.number_input("Input 1", value=0.5)
x2 = st.number_input("Input 2", value=0.5)

inputs = [x1, x2]

# ---------------------------
# Single Neuron UI
# ---------------------------
if model == "Single Neuron":

    st.subheader("Weights and Bias")

    w1 = st.number_input("Weight 1", value=0.2)
    w2 = st.number_input("Weight 2", value=0.8)
    bias = st.number_input("Bias", value=0.1)

    if st.button("Compute"):
        response = requests.post(
            "http://127.0.0.1:5000/single",
            json={
                "inputs": inputs,
                "weights": [w1, w2],
                "bias": bias
            }
        )

        result = response.json()
        st.success(f"Output: {result['output']}")


# ---------------------------
# 3 Neuron Layer UI
# ---------------------------
if model == "1 Layer (3 Neurons)":

    st.subheader("Neuron 1")
    w11 = st.number_input("W11", value=0.2)
    w12 = st.number_input("W12", value=0.3)
    b1 = st.number_input("Bias 1", value=0.1)

    st.subheader("Neuron 2")
    w21 = st.number_input("W21", value=0.4)
    w22 = st.number_input("W22", value=0.5)
    b2 = st.number_input("Bias 2", value=0.2)

    st.subheader("Neuron 3")
    w31 = st.number_input("W31", value=0.6)
    w32 = st.number_input("W32", value=0.7)
    b3 = st.number_input("Bias 3", value=0.3)

    if st.button("Compute Layer Output"):

        response = requests.post(
            "http://127.0.0.1:5000/layer",
            json={
                "inputs": inputs,
                "weights": [
                    [w11, w12],
                    [w21, w22],
                    [w31, w32]
                ],
                "biases": [b1, b2, b3]
            }
        )

        result = response.json()
        st.success(f"Outputs: {result['outputs']}")
