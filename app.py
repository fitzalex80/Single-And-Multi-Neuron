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
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()

    def forward(self, x):
        z = np.dot(x, self.weights) + self.bias
        return sigmoid(z)

# ----------------------------
# 1 Layer with 3 Neurons
# ----------------------------
class ThreeNeuronLayer:
    def __init__(self, input_size):
        self.weights = np.random.randn(3, input_size)
        self.bias = np.random.randn(3)

    def forward(self, x):
        z = np.dot(self.weights, x) + self.bias
        return sigmoid(z)


# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Neural Network Demo (Classical Implementation)")

st.write("## Enter Two Inputs")

x1 = st.number_input("Input 1", value=0.0)
x2 = st.number_input("Input 2", value=0.0)

inputs = np.array([x1, x2])

# Create models
single_neuron = SingleNeuron(2)
three_layer = ThreeNeuronLayer(2)

if st.button("Run Single Neuron"):
    output = single_neuron.forward(inputs)
    st.success(f"Single Neuron Output: {output}")

if st.button("Run 3-Neuron Layer"):
    output = three_layer.forward(inputs)
    st.success(f"3 Neuron Layer Output: {output}")