# backend/app.py

from flask import Flask, request, jsonify
import math

app = Flask(__name__)

# ---------------------------
# Activation Function
# ---------------------------
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# ---------------------------
# Single Neuron
# ---------------------------
def single_neuron(inputs, weights, bias):
    weighted_sum = 0
    for i in range(len(inputs)):
        weighted_sum += inputs[i] * weights[i]
    
    weighted_sum += bias
    return sigmoid(weighted_sum)


# ---------------------------
# 1 Layer with 3 Neurons
# ---------------------------
def one_layer_three_neurons(inputs, weights, biases):
    outputs = []

    for neuron in range(3):
        weighted_sum = 0
        for i in range(len(inputs)):
            weighted_sum += inputs[i] * weights[neuron][i]

        weighted_sum += biases[neuron]
        outputs.append(sigmoid(weighted_sum))

    return outputs


# ---------------------------
# API for Single Neuron
# ---------------------------
@app.route("/single", methods=["POST"])
def single():
    data = request.json
    inputs = data["inputs"]
    weights = data["weights"]
    bias = data["bias"]

    result = single_neuron(inputs, weights, bias)
    return jsonify({"output": result})


# ---------------------------
# API for 3 Neuron Layer
# ---------------------------
@app.route("/layer", methods=["POST"])
def layer():
    data = request.json
    inputs = data["inputs"]
    weights = data["weights"]
    biases = data["biases"]

    result = one_layer_three_neurons(inputs, weights, biases)
    return jsonify({"outputs": result})


if __name__ == "__main__":
    app.run(debug=True)
