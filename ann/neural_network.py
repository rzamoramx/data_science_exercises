""" Neural network of three layers input, hidden and output, uses logistic sigmoid as activation function """
import numpy
import scipy.special  # this is for import sigmoid function (expit)


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        self.l_rate = learning_rate

        # Link weight matrices, wih and who weights inside the arrays are w_i_j, where
        # link is from node i to node j in the next layer
        # w11 w21, w12 w22 etc
        # self.wih = (numpy.random.rand(self.h_nodes, self.i_nodes) - 0.5)
        # self.who = (numpy.random.rand(self.o_nodes, self.h_nodes) - 0.5)

        # Better refinement weights
        self.wih = numpy.random.normal(0.0, pow(self.i_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.who = numpy.random.normal(0.0, pow(self.h_nodes, -0.5), (self.o_nodes, self.h_nodes))

        # this is the sigmoid function activation
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        # converts inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Calculate the signal emerging from signal output layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signal emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # the error is the (target - actual)
        output_errors = targets - final_outputs

        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.l_rate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.l_rate * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

    def query(self, inputs_list):
        # convert input list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Calculate the signal emerging from signal output layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signal emerging from final output layer
        return self.activation_function(final_inputs)
