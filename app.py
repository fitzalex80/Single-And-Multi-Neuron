import streamlit as st
import numpy as np

# ----------------------------
# Activation Function
# ----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ----------------------------
# Single Neuron
# ----------------------------
class SingleNeuron:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias

    def forward(self, x):
        z = np.dot(x, self.weights) + self.bias
        return sigmoid(z)

# ----------------------------
# 1 Layer with 3 Neurons
# ----------------------------
class ThreeNeuronLayer:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = np.array(bias)

    def forward(self, x):
        z = np.dot(self.weights, x) + self.bias
        return sigmoid(z)


# =============================
# STREAMLIT UI
# =============================

st.title("Neural Network (Classical Implementation)")

# Select Model Type
model_type = st.radio(
    "Select Model Type:",
    ("Single Neuron", "1 Layer with 3 Neurons")
)

st.header("Enter Inputs")
x1 = st.number_input("Input 1", value=0.0)
x2 = st.number_input("Input 2", value=0.0)
inputs = np.array([x1, x2])


# =============================
# SINGLE NEURON INPUTS
# =============================
if model_type == "Single Neuron":
    st.header("Weights and Bias (Single Neuron)")
    
    w1 = st.number_input("Weight 1", value=0.5)
    w2 = st.number_input("Weight 2", value=0.5)
    b = st.number_input("Bias", value=0.0)


# =============================
# THREE NEURON LAYER INPUTS
# =============================
else:
    st.header("Weights and Bias (3 Neuron Layer)")

    st.subheader("Neuron 1")
    w11 = st.number_input("W11", value=0.2)
    w12 = st.number_input("W12", value=0.2)
    b1 = st.number_input("Bias 1", value=0.0)

    st.subheader("Neuron 2")
    w21 = st.number_input("W21", value=0.3)
    w22 = st.number_input("W22", value=0.3)
    b2 = st.number_input("Bias 2", value=0.0)

    st.subheader("Neuron 3")
    w31 = st.number_input("W31", value=0.4)
    w32 = st.number_input("W32", value=0.4)
    b3 = st.number_input("Bias 3", value=0.0)


# =============================
# SINGLE BUTTON TO RUN MODEL
# =============================
if st.button("Find Output"):

    if model_type == "Single Neuron":
        neuron = SingleNeuron([w1, w2], b)
        output = neuron.forward(inputs)

        st.write("### Calculation")
        st.write(f"z = (w1*x1 + w2*x2) + b")
        st.write(f"Output = Sigmoid(z)")
        st.success(f"Final Output: {output}")

    else:
        weights = [
            [w11, w12],
            [w21, w22],
            [w31, w32]
        ]
        biases = [b1, b2, b3]

        layer = ThreeNeuronLayer(weights, biases)
        output = layer.forward(inputs)

        st.write("### Calculation")
        st.write("Each neuron computes: z = w·x + b")
        st.write("Then applies Sigmoid activation")

        st.success(f"Final Outputs: {output}")
