import random
from typing import Callable
import numpy as np

_FUNCTION = 0
_DERIVATIVE = 1

class NeuralNetwork:
    def __init__(self, structure : list[tuple[int, tuple[Callable, Callable]]], start_interval : tuple[int, int]):
        self._weights : list[list[list[int]]] = []
        self._gradient : list[list[list[int]]] = []
        self._in : list[list[float]] = []
        self._out : list[list[float]] = []
        self._error : list[list[float]] = []
        self._layers : list[int] = []
        self._activation_functions : list[tuple[Callable, Callable]] = [] # Tuples containing the activation function and its derivative

        for k, structure_tuple in enumerate(structure):
            layer_neurons, activation_function = structure_tuple

            if k < len(structure) - 1:
                next_layer_neurons, _ = structure[k + 1]
                self._weights.append([])
                self._gradient.append([])
                for i in range(layer_neurons + 1):
                    self._weights[k].append([])
                    self._gradient[k].append([])
                    for _ in range(next_layer_neurons + 1):
                        self._weights[k][i].append(random.random() * random.randint(start_interval[0], start_interval[1]))
                        self._gradient[k][i].append(0)

            self._in.append([])
            self._out.append([])
            self._error.append([])
            for _ in range(layer_neurons + 1):
                self._in[k].append(0)
                self._out[k].append(0)
                self._error[k].append(0)

            self._layers.append(layer_neurons)
            self._activation_functions.append(activation_function)
        
    def fit(self, X, y, learning_rate : float = 1, error_threshold : float = 0.2):    
        while True:
            self._initialize_gradient()        
            mean_error = 0
            for example_input, example_output in zip(X, y):
                self._feed_forward(example_input)
                self._back_propagate(example_output)

                prediction = np.array(self._out[-1][1:])
                truth = np.array(example_output)
                difference = np.subtract(truth, prediction)
                mean_error += np.linalg.norm(difference)
            
            mean_error /= len(X)
            if mean_error < error_threshold:
                return
            
            for k, _ in enumerate(self._weights):
                for i, _ in enumerate(self._weights[k]):
                    for j, _ in enumerate(self._weights[k][i]):
                        self._weights[k][i][j] = self._weights[k][i][j] - learning_rate * self._gradient[k][i][j]

    def predict(self, X):
        y = []
        for input in X:
            self._feed_forward(input)
            y.append(self._out[-1][1:])
        return y

    def _feed_forward(self, input : list[float]):
        self._out[0][0] = 1
        for i, value in enumerate(input):
            self._out[0][i + 1] = value

        for k, neurons in enumerate(self._layers[1:]):
            k += 1
            for i in range(neurons + 1):
                if i == 0:
                    self._in[k][i] = 0
                    self._out[k][i] = 1
                else:
                    incoming = np.dot(
                        np.array(self._out[k - 1]), 
                        np.array([row[i] for row in self._weights[k - 1]]))
                    
                    outgoing = self._activation_functions[k][_FUNCTION](incoming)

                    self._in[k][i] = incoming
                    self._out[k][i] = outgoing

    def _back_propagate(self, output : list[float]):
        output = output[:] 
        output.insert(0, 1)

        for k in range(len(self._layers) - 1, 0, -1):
            for i in range(1, len(self._error[k])):
                local_derivative = self._activation_functions[k][_DERIVATIVE](self._in[k][i])
            
                if k == len(self._layers) - 1:
                    self._error[k][i] = -2 * (output[i] - self._out[k][i]) * local_derivative
                else:
                    incoming_errors = 0
                    for j, incoming_error in enumerate(self._error[k + 1][1:]):
                        incoming_errors += incoming_error * self._weights[k][i][j]
                    self._error[k][i] = incoming_errors * local_derivative

                for j in range(self._layers[k - 1]):
                    self._gradient[k - 1][j][i] += self._error[k][i] * self._out[k - 1][j]

    def _initialize_gradient(self):
        for k, _ in enumerate(self._gradient):
            for i, _ in enumerate(self._gradient[k]):
                for j, _ in enumerate(self._gradient[k][i]):
                    self._gradient[k][i][j] = 0
