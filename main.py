import random
import numpy as np
from math import exp
from matplotlib import pyplot as plt


class NeuralNetwork:

    def __init__(self, input_size, output_size, learning_rate=3, max_error=0.01, bias=0):
        self.weights = []
        self.input_size = input_size
        self.output_size = output_size
        self.output_layer = np.zeros(output_size)
        self.learning_rate = learning_rate
        self.max_error = max_error
        self.bias = bias
        self.current_error = 0
        self.randomize_weights()
        # self.activation_funcs = []
        # self.activation_funcs = {
        #     "sigmoid": sigmoid, "linear": linear,
        #     "relu": relu, "uni_sigmoid": uni_sigmoid}

    def randomize_weights(self):
        for idx in range(self.input_size + 1):
            self.weights.append([random.uniform(-1, 1) for idx in range(self.output_size)])

    def compute_outputs(self, input_layer):
        output_value_holder = 0
        for out_idx in range(self.output_size):
            for in_idx in range(len(input_layer)):
                output_value_holder += input_layer[in_idx] * self.weights[in_idx][out_idx]
            output_value_holder += self.bias * self.weights[len(input_layer)][out_idx]
            self.output_layer[out_idx] = self.activation(output_value_holder)

    # relu is default activation
    def activation(self, value):
        return 1 if value >= 0 else 0

    def propagate_input(self, input_layer, out_target=0, update_weights=True):
        self.compute_outputs(input_layer)
        if update_weights:
            self.update_weights(input_layer, out_target)

    def update_weights(self, input_layer, out_target):
        for out_idx in range(self.output_size):
            for in_idx in range(len(input_layer)):
                outp = self.output_layer[out_idx]
                basic_error = out_target - outp
                self.weights[in_idx][out_idx] = self.weights[in_idx][out_idx] + self.learning_rate * basic_error * (
                        1 - outp) * input_layer[in_idx] * outp

    def compute_error(self, out_target):
        error = 0
        idx = 0
        for outp in self.output_layer:
            error += pow(out_target - outp, 2) / 2
            idx += 1
        return error


class Linear(NeuralNetwork):
    def __init__(self, input_size, output_size, learning_rate=3, max_error=0.01, bias=0, a_factor=1):
        NeuralNetwork.__init__(self, input_size, output_size, learning_rate, max_error, bias)
        self.a_factor = a_factor

    def activation(self, value):
        # print("linear here")
        return self.a_factor * value


class Sigmoid(NeuralNetwork):
    def __init__(self, input_size, output_size, learning_rate=3, max_error=0.01, bias=0):
        NeuralNetwork.__init__(self, input_size, output_size, learning_rate, max_error, bias)

    def activation(self, value):
        if abs(value) >= 710:
            return 0
        return 1 / (1 - exp(value * (-1)))


class ReLu(NeuralNetwork):
    def __init__(self, input_size, output_size, learning_rate=3, max_error=0.01, bias=0):
        NeuralNetwork.__init__(self, input_size, output_size, learning_rate, max_error, bias)

    def activation(self, value):
        return 1 if value >= 0 else 0


class UniSigmoid(NeuralNetwork):
    def __init__(self, input_size, output_size, learning_rate=3, max_error=0.01, bias=0, l_factor=1):
        NeuralNetwork.__init__(self, input_size, output_size, learning_rate, max_error, bias)
        self.l_factor = l_factor

    def activation(self, value):
        if abs(value) >= 710:
            return 0
        return 1 / (1 + exp(-value * self.l_factor))


def fill_0_or_conv_to_np(arr=[]):
    arr = np.array(arr)
    if arr.size == 0:
        arr = np.array([[-2, -2], [-2, -2]])
    return arr


# 2 input neuron
y_values = []
x_values = []
points = []
x_point = -2.0
y_point = -2.0
up_bound = 2.0
resolution = 0.01

while not x_point - 0.1 >= up_bound:
    x_values.append(x_point)
    x_point += resolution
while not y_point - 0.1 >= up_bound:
    y_values.append(y_point)
    y_point += resolution
for x in x_values:
    for y in y_values:
        points.append([x, y])

# for j in range(0, 2):
#     blue = []
#     red = []
#     yellow = []
#     green = []
#     navy_blue = []
#     color_arrays = [blue, red, yellow, green, navy_blue]
#
#     print(j)
#     relu = ReLu(2, 1, bias=j)
#     for x in points:
#         relu.propagate_input(input_layer=x, update_weights=False)
#         for outp in relu.output_layer:
#             if outp == 1:
#                 color_arrays[1].append(x)
#             elif outp == 0:
#                 color_arrays[0].append(x)
#         relu.randomize_weights()
#
#     for cl_arr_idx in range(len(color_arrays)):
#         color_arrays[cl_arr_idx] = fill_0_or_conv_to_np(color_arrays[cl_arr_idx])
#
#     figure = plt.Figure()
#     plt.plot(color_arrays[0][:, 0], color_arrays[0][:, 1], 'o', c='blue')
#     plt.plot(color_arrays[1][:, 0], color_arrays[1][:, 1], 'o', c='red')
#     figure.suptitle("ReLu")
#     plt.show()
#
#     for cl_arr_idx in range(len(color_arrays)):
#         color_arrays[cl_arr_idx] = []
#
#     linear = Linear(2, 1, a_factor=2, bias=j)
#     for x in points:
#         linear.propagate_input(input_layer=x, update_weights=False)
#         for outp in linear.output_layer:
#             if outp < -2:
#                 color_arrays[4].append(x)
#             elif -2 <= outp <= 0:
#                 color_arrays[0].append(x)
#             elif 0 <= outp <= 2:
#                 color_arrays[3].append(x)
#             elif outp > 2:
#                 color_arrays[1].append(x)
#         linear.randomize_weights()
#
#     for cl_arr_idx in range(len(color_arrays)):
#         color_arrays[cl_arr_idx] = fill_0_or_conv_to_np(color_arrays[cl_arr_idx])
#
#     figure = plt.Figure()
#     plt.plot(color_arrays[0][:, 0], color_arrays[0][:, 1], 'o', c='blue')
#     plt.plot(color_arrays[1][:, 0], color_arrays[1][:, 1], 'o', c='red')
#     plt.plot(color_arrays[3][:, 0], color_arrays[3][:, 1], 'o', c='green')
#     plt.plot(color_arrays[4][:, 0], color_arrays[4][:, 1], 'o', c='m')
#     figure.suptitle("linear")
#     plt.show()
#
#     for cl_arr_idx in range(len(color_arrays)):
#         color_arrays[cl_arr_idx] = []
#
#     sigm = UniSigmoid(2, 1, bias=j, l_factor=2)
#     for x in points:
#         sigm.propagate_input(input_layer=x, update_weights=False)
#         for outp in sigm.output_layer:
#             if 0 <= outp <= 0.25:
#                 color_arrays[2].append(x)
#             elif 0.25 < outp <= 0.5:
#                 color_arrays[0].append(x)
#             elif 0.5 < outp <= 0.75:
#                 color_arrays[3].append(x)
#             elif 0.75 < outp <= 1:
#                 color_arrays[1].append(x)
#         # sigm.randomize_weights()
#
#     for cl_arr_idx in range(len(color_arrays)):
#         color_arrays[cl_arr_idx] = fill_0_or_conv_to_np(color_arrays[cl_arr_idx])
#
#     figure = plt.Figure()
#     plt.plot(color_arrays[2][:, 0], color_arrays[2][:, 1], 'o', c='y')
#     plt.plot(color_arrays[0][:, 0], color_arrays[0][:, 1], 'o', c='blue')
#     plt.plot(color_arrays[3][:, 0], color_arrays[3][:, 1], 'o', c='green')
#     plt.plot(color_arrays[1][:, 0], color_arrays[1][:, 1], 'o', c='red')
#     figure.suptitle("linear")
#     plt.show()

# 2 layer, 2 input, 1 output
for j in range(0, 2):
    print(j)
    blue = []
    red = []
    yellow = []
    green = []
    navy_blue = []
    color_arrays = [blue, red, yellow, green, navy_blue]

    relu1 = ReLu(2, 2, bias=j)
    relu2 = ReLu(2, 1, bias=j)
    for x in points:
        relu1.propagate_input(input_layer=x, update_weights=False)
        relu2.propagate_input(input_layer=relu1.output_layer, update_weights=False)
        for outp in relu2.output_layer:
            if outp == 1:
                color_arrays[1].append(x)
            elif outp == 0:
                color_arrays[0].append(x)
        relu1.randomize_weights()
        relu2.randomize_weights()

    for cl_arr_idx in range(len(color_arrays)):
        color_arrays[cl_arr_idx] = fill_0_or_conv_to_np(color_arrays[cl_arr_idx])

    figure = plt.Figure()
    plt.plot(color_arrays[0][:, 0], color_arrays[0][:, 1], 'o', c='blue')
    plt.plot(color_arrays[1][:, 0], color_arrays[1][:, 1], 'o', c='red')
    figure.suptitle("ReLu")
    plt.show()

    for cl_arr_idx in range(len(color_arrays)):
        color_arrays[cl_arr_idx] = []

    linear1 = Linear(2, 2, a_factor=2, bias=j)
    linear2 = Linear(2, 1, a_factor=1, bias=j)
    for x in points:
        linear1.propagate_input(input_layer=x, update_weights=False)
        linear2.propagate_input(input_layer=linear1.output_layer, update_weights=False)

        for outp in linear2.output_layer:
            if outp < -2:
                color_arrays[4].append(x)
            elif -2 <= outp <= 0:
                color_arrays[0].append(x)
            elif 0 <= outp <= 2:
                color_arrays[3].append(x)
            elif outp > 2:
                color_arrays[1].append(x)
        linear1.randomize_weights()
        linear2.randomize_weights()

    for cl_arr_idx in range(len(color_arrays)):
        color_arrays[cl_arr_idx] = fill_0_or_conv_to_np(color_arrays[cl_arr_idx])

    figure = plt.Figure()
    plt.plot(color_arrays[0][:, 0], color_arrays[0][:, 1], 'o', c='blue')
    plt.plot(color_arrays[1][:, 0], color_arrays[1][:, 1], 'o', c='red')
    plt.plot(color_arrays[3][:, 0], color_arrays[3][:, 1], 'o', c='green')
    plt.plot(color_arrays[4][:, 0], color_arrays[4][:, 1], 'o', c='m')
    figure.suptitle("linear")
    plt.show()

    for cl_arr_idx in range(len(color_arrays)):
        color_arrays[cl_arr_idx] = []

    sigm1 = UniSigmoid(2, 2, bias=j, l_factor=2)
    sigm2 = UniSigmoid(2, 1, bias=j, l_factor=2)
    for x in points:
        sigm1.propagate_input(input_layer=x, update_weights=False)
        sigm2.propagate_input(input_layer=sigm1.output_layer, update_weights=False)
        for outp in sigm2.output_layer:
            if 0 <= outp <= 0.25:
                color_arrays[2].append(x)
            elif 0.25 < outp <= 0.5:
                color_arrays[0].append(x)
            elif 0.5 < outp <= 0.75:
                color_arrays[3].append(x)
            elif 0.75 < outp <= 1:
                color_arrays[1].append(x)
        sigm1.randomize_weights()
        sigm2.randomize_weights()

    for cl_arr_idx in range(len(color_arrays)):
        color_arrays[cl_arr_idx] = fill_0_or_conv_to_np(color_arrays[cl_arr_idx])

    figure = plt.Figure()
    plt.plot(color_arrays[2][:, 0], color_arrays[2][:, 1], 'o', c='y')
    plt.plot(color_arrays[0][:, 0], color_arrays[0][:, 1], 'o', c='blue')
    plt.plot(color_arrays[3][:, 0], color_arrays[3][:, 1], 'o', c='green')
    plt.plot(color_arrays[1][:, 0], color_arrays[1][:, 1], 'o', c='red')
    figure.suptitle("linear")
    plt.show()

