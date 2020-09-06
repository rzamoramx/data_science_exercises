""" Exercises for ANN """
import numpy
from ann.neural_network import NeuralNetwork
import matplotlib.pyplot as plt
from skimage import transform, io


def main():
    # Sampling inputs and outputs

    #view_example_of_training_data()
    #desired_target_test = 9  # specific position on the test file
    #view_example_of_test_data(desired_target_test)
    #view_my_own_handwritten_number("nueve")

    # ANN Operating

    # the matrix of the number image (hand written) in this case 28x28 = 784
    input_nodes = 784
    # the same length of training data set
    hidden_nodes = 200
    # Every possible answer in this case 0-9 numbers
    output_nodes = 10

    learning_rate = 0.1

    epochs = 5

    # Init ANN
    ann = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    operates_ann(ann, output_nodes, epochs, False, True, 1)

    plt.show()  # for show the plotted images


def operates_ann(ann: NeuralNetwork, output_nodes: int, epochs: int, my_handwritten=False, large_set=False, label_backquery=-1):
    """
    Operates de ANN
    :param ann: instance of ANN
    :param output_nodes: number of nodes for ANN
    :param epochs: number of epochs
    :param my_handwritten: if we wants to operate with own handwritten numbers
    :param large_set: if we wants to train and test the ANN with large set of training data located in resources
    :param label_backquery: -1 if we don't want a backward query, 0 to 9 (numbers) for backward query to ANN
    :return: nothing
    """
    # START TRAINING
    training_list = get_training(large_set)

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
    test_list = get_test(my_handwritten, large_set)

    # test all
    for record in test_list:
        if my_handwritten is True:
            all_values = record
        else:
            # split the element of the list
            all_values = record.split(',')

        # correct answer
        correct_label = int(all_values[0])
        print(f'correct label {repr(correct_label)}')
        # scale and shift the inputs
        str_to_int = numpy.asfarray(all_values[1:])  # converts strings into number
        scaled_test = (str_to_int / 255.0 * 0.99) + 0.01

        # Query
        outputs = ann.query(scaled_test)
        # the highest value is the answer of the ann
        label = numpy.argmax(outputs)
        print(f'ann answer {label}')

        if (label == correct_label):
            score_card.append(1)
        else:
            score_card.append(0)

    # Calculate the performance score
    print(score_card)
    score_a = numpy.asarray(score_card)
    print(f'performance percentage of accuracy {score_a.sum()/score_a.size}')

    # Prepare outputs avoiding zeros and ones
    # output nodes is 10, because need 10 possible responses
    #onodes = 10
    #targets = numpy.zeros(onodes) + 0.01
    #targets[int(all_values[0])] = 0.99  # targets are correct answers so is equal to one, but can't use that number

    # Backquery
    if label_backquery > -1:
        # create the output signals for this label
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[label_backquery] = 0.99
        print(targets)

        # get image data
        image_data = ann.backquery(targets)

        # plot image data
        plt.figure()
        plt.imshow(image_data.reshape(28,28), cmap='Greys', interpolation='None')


def view_my_own_handwritten_number(my_number: str):
    """
    Only for check own handwritten images located in resources/my_handwritten
    :param my_number: number to show
    :return:
    """
    img = io.imread("resources/my_handwritten/"+my_number+".png", as_gray=True)
    img_data = transform.resize(img, (28,28), mode='symmetric', preserve_range=True)

    img_data = (img_data / 255.0 * 0.99) + 0.01  # re-escala la data a valores decimales de entre 0.01 y 1.0

    print(img_data)

    plt.figure()
    plt.imshow(img_data, cmap='Greys', interpolation='None')


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


def get_training(large_set=True) -> list:
    if large_set:
        training_file = open("resources/mnist_train.csv", 'r')
    else:
        training_file = open("resources/mnist_train_100.csv", 'r')
    training_list = training_file.readlines()
    training_file.close()
    return training_list


def get_test(my_own_handwritten=False, large_set=True) -> list:
    if my_own_handwritten:
        return get_my_handwritten()

    if large_set:
        test_file = open("resources/mnist_test.csv", 'r')
    else:
        test_file = open("resources/mnist_test_10.csv", 'r')
    test_list = test_file.readlines()
    test_file.close()
    return test_list


def get_my_handwritten() -> list:
    what_numbers = {"uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "ocho": 8, "nueve": 9}
    list_data = []
    for key in what_numbers:
        img = io.imread("resources/my_handwritten/"+key+".png", as_gray=True)
        img_data = transform.resize(img, (28,28), mode='symmetric', preserve_range=True)

        img_data = (img_data / 255.0 * 0.99) + 0.01  # re-escala la data a valores decimales de entre 0.01 y 1.0
        img_data = numpy.insert(img_data, 0, what_numbers[key])
        #img_data.insert(0, what_numbers[key])
        list_data.append(img_data)
    return list_data


if __name__ == '__main__':
    main()
