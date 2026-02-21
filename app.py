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
        self.weights = np.array(weights)   # shape (3, input_size)
        self.bias = np.array(bias)         # 3 biases

    def forward(self, x):
        z = np.dot(self.weights, x) + self.bias
        return sigmoid(z)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Neural Network Demo (User Defined Weights & Bias)")

st.header("Enter Inputs")
x1 = st.number_input("Input 1", value=0.0)
x2 = st.number_input("Input 2", value=0.0)

inputs = np.array([x1, x2])

# ===============================
# SINGLE NEURON SECTION
# ===============================
st.header("Single Neuron")

w1 = st.number_input("Weight 1", value=0.5)
w2 = st.number_input("Weight 2", value=0.5)
b_single = st.number_input("Bias (Single Neuron)", value=0.0)

if st.button("Run Single Neuron"):
    neuron = SingleNeuron([w1, w2], b_single)
    output = neuron.forward(inputs)
    st.success(f"Output: {output}")

# ===============================
# THREE NEURON LAYER SECTION
# ===============================
st.header("1 Layer with 3 Neurons")

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

if st.button("Run 3 Neuron Layer"):
    weights = [
        [w11, w12],
        [w21, w22],
        [w31, w32]
    ]
    biases = [b1, b2, b3]

    layer = ThreeNeuronLayer(weights, biases)
    output = layer.forward(inputs)
    st.success(f"Outputs: {output}")
