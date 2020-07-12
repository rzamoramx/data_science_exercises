""" Exercises for ANN """
import numpy
from ann.neural_network import NeuralNetwork
import matplotlib.pyplot as plt


def main():
    # the matrix of the number image (hand written) in this case 28x28 = 784
    input_nodes = 784
    # the same length of training data set
    hidden_nodes = 200
    # Every possible answer in this case 0-9 numbers
    output_nodes = 10

    learning_rate = 0.1

    desired_target_test = 9  # the position on the test file

    epochs = 5

    # Init ANN
    ann = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    #view_example_of_training_data()
    #view_example_of_test_data(desired_target_test)
    operates_ann(ann, output_nodes, epochs)

    plt.show()  # for show the plotted images


def operates_ann(ann: NeuralNetwork, output_nodes: int, epochs: int):
    # START TRAINING
    training_list = get_training()

    score_card = []

    for _ in range(epochs):
        # for every case in training set
        for record in training_list:
            all_values = record.split(',')
            # scale inputs
            # first convert str to int, except target label [1:] and scale
            inputs = (numpy.asfarray(all_values[1:]) / 250.0 * 0.99) + 0.01
            # create the target output (all 0.01, except the desired label witch is 0.99)
            targets = numpy.zeros(output_nodes) + 0.01
            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99
            ann.train(inputs, targets)

    # START TESTING
    test_list = get_test()

    # test all
    for record in test_list:
        # split the element of the list
        all_values = record.split(',')
        # correct answer
        correct_label = int(all_values[0])
        print(f'correct label {correct_label}')
        # scale and shift the inputs
        str_to_int = numpy.asfarray(all_values[1:])  # converts strings into number
        scaled_test = (str_to_int / 255.0 * 0.99) + 0.01
        # make query
        outputs = ann.query(scaled_test)
        # the highest value is the answer of the ann
        label = numpy.argmax(outputs)
        print(f'ann answer {label}')

        if (label == correct_label):
            score_card.append(1)
        else:
            score_card.append(0)

    print(score_card)
    score_a = numpy.asarray(score_card)
    print(f'performance percentage of accuracy {score_a.sum()/score_a.size}')

    # Prepare outputs avoiding zeros and ones
    # output nodes is 10, because need 10 possible responses
    #onodes = 10
    #targets = numpy.zeros(onodes) + 0.01
    #targets[int(all_values[0])] = 0.99  # targets are correct answers so is equal to one, but can't use that number


def view_example_of_test_data(desired_target_test: int):
    # Read data from test file
    test_list = get_test()

    # to get a sample of this training data set
    all_values = test_list[desired_target_test].split(',')
    # print the label
    print(all_values[0])
    str_to_int = numpy.asfarray(all_values[1:])  # converts strings into number
    image_array = str_to_int.reshape((28, 28))  # put numbers in a matrix of 28x28
    plt.figure()
    plt.imshow(image_array, cmap='Greys', interpolation='None')


def view_example_of_training_data():
    # Read data from train file
    training_list = get_training()

    # to get a sample of this training data set
    all_values = training_list[1].split(',')
    str_to_int = numpy.asfarray(all_values[1:])  # converts strings into number
    image_array = str_to_int.reshape((28, 28))  # put numbers in a matrix of 28x28
    plt.figure()
    plt.imshow(image_array, cmap='Greys', interpolation='None')

    # example of scale input data, needed for sigmoid function that operates between 0.01 and 0.99 (0-1)
    # 0 and 1 cannot use because this kill learning capacities of the ann
    scaled_input = (str_to_int / 255.0 * 0.99) + 0.01
    print(scaled_input)


def get_training() -> list:
    #training_file = open("resources/mnist_train_100.csv", 'r')
    training_file = open("resources/mnist_train.csv", 'r')
    training_list = training_file.readlines()
    training_file.close()
    return training_list


def get_test() -> list:
    #test_file = open("resources/mnist_test_10.csv", 'r')
    test_file = open("resources/mnist_test.csv", 'r')
    test_list = test_file.readlines()
    test_file.close()
    return test_list


if __name__ == '__main__':
    main()


