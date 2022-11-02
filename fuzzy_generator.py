import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.metrics import mean_squared_error


def create_tables(x1, x2):  # index the ruletables by prim_table.loc[x1_label][x2_label]
    """
    Function to create the primary and secondary ruletable, filled with NaNs
    :param x1: labels of input 1
    :param x2: labels of input 2
    :return: primary ruletable, secondary ruletable, filled with NaNs, respectively
    """
    prim_table = pd.DataFrame(columns=x1, index=x2)
    sec_table = pd.DataFrame(columns=x1, index=x2)
    return prim_table, sec_table


class FuzzyPartition:
    """
    class to create a fuzzy partition
    """
    def __init__(self, mu_sigma_list, labels, name):
        """
        constructor to create an instance of a fuzzy partition
        :param mu_sigma_list: list of lists, containing the mus and sigmas of the gaussians that represent the fuzzy set
        :param labels: labels of the gaussians
        :param name: name of the entire partition
        """
        self.fuzzy_partition = self.generate_fuzzy_partition(mu_sigma_list)
        self.mu_sigma_list = mu_sigma_list
        self.labels = labels
        self.name = name

    def create_gaussian(self, mu, sigma):
        """
        function to create a gaussian distribution on the basis of a mu and sigma value
        :param mu: mean of distribution
        :param sigma: standard deviation of distribution
        :return: function of gaussian distribution with mu and sigma as the relevant values
        """
        return lambda x: np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

    def generate_fuzzy_partition(self, mu_sigma_list):
        """
        function to create a fuzzy partition, constisting of multiple gaussian distributions
        :param mu_sigma_list: list of lists, containing the mus and sigmas of the fuzzy partition
        :return: list of gaussians representing the fuzzy set
        """
        gaussians = []
        for sublist in mu_sigma_list:
            mu = sublist[0]
            sigma = sublist[1]
            gaussians.append(self.create_gaussian(mu, sigma))
        return gaussians


def calc_b(x, a, partition_in_1, partition_in_2):
    """
    function to calculate b, based on the input and the output
    :param x: list with output in last position and inputs as other elements
    :param a: positive constant
    :param partition_in_1: list of membership functions of input 1 in the form of gaussian distributions
    :param partition_in_2: list of membership functions of input 2 in the form of gaussian distributions
    :return: b
    """
    x_p = x[0:2]  # get input values
    y_p = x[-1]  # get output value
    weights = [calc_compatibility_degree(x_p[0], partition_in_1) ** a,
               calc_compatibility_degree(x_p[1], partition_in_2) ** a]
    y_p = np.array(y_p)
    weights = np.array(weights)
    #b = sum(weights * y_p) / sum(weights)
    b = sum(weights * y_p) / sum(weights)
    #print(f'weights.shape: {weights.shape}, y_p.shape: {y_p.shape}')
    return b


def calc_compatibility_degree(x, partition):
    """
    function to calculate the compability degree of a certain x with a fuzzy partition
    :param x: list of input values
    :param partition: list of membership functions in the form of gaussian distributions
    :return: degree of compatibility of input vector x to the fuzzy if-then-rule
    """
    membership_value = membership_function(partition, x)
    comp_deg = 1
    for val in membership_value:
        comp_deg *= val
    return comp_deg


def membership_function(partition, x):
    """
    function to calculate the y-value of a certain x under each of the gaussian distributions of a fuzzy partition
    :param partition: list of membership functions in the form of gaussian distributions
    :param x: input value
    :return: list of membership values of x with each of the gaussians
    """
    return [gaussian(x) for gaussian in partition]


def sort_by_membership_value(partition, x):
    """
    function to get the membership values sorted and the corresponding indices (in order to be able to determine the
    corresponding label)
    :param partition: list of membership functions in the form of gaussian distributions
    :param x: input value
    :return: list of indices, sorted by increasing membership value, and list of sorted membership values (increasing)
    """
    membership_values = membership_function(partition, x)
    sorted_indices = np.argsort(membership_values)
    sorted_membership_values = np.sort(membership_values)
    return sorted_indices, sorted_membership_values


def get_b_values(partition, labels, b):
    """
    function to determine b_star (label of gaussian with the highest value for b), b_star_star (label of gaussian with
    the second-highest value for b) and the respective degrees of certainty
    :param partition: list of membership functions in the form of gaussian distributions
    :param labels: names of the fuzzy sets in the partition
    :param b: calculated b-value
    :return:
    """
    sorted_labels, sorted_values = sort_by_membership_value(partition, b)  # in ascending order
    b_star = labels[sorted_labels[-1]]
    b_star_star = labels[sorted_labels[-2]]
    b_star_cf = sorted_values[-1]
    b_star_star_cf = sorted_values[-2]
    return b_star, b_star_star, b_star_cf, b_star_star_cf


def fill_table(partition_in_1, partition_in_2, partition_out, x, prim_table, sec_table, a):
    """
    function to fill the primary and secondary rule table
    :param partition_in_1: fuzzy partition on input 1
    :param partition_in_2: fuzzy partition on input 2
    :param partition_out: fuzzy partition on output
    :param x: input
    :param prim_table: primary rule table
    :param sec_table: secondary rule table
    :param a: alpha value
    :return: primary and secondary rule table
    """
    b = calc_b(x, a, partition_in_1.fuzzy_partition, partition_in_2.fuzzy_partition)
    b_star, b_star_star, b_star_cf, b_star_star_cf = \
        get_b_values(partition_out.fuzzy_partition,
                     partition_out.labels, b)  # primary and secondary label of the output
    prim_label_1, sec_label_1, prim_1_cf, sec_1_cf = \
        get_b_values(partition_in_1.fuzzy_partition,
                     partition_in_1.labels, x[0])  # primary and secondary label on input 1
    prim_label_2, sec_label_2, prim_2_cf, sec_2_cf = \
        get_b_values(partition_in_2.fuzzy_partition,
                     partition_in_2.labels, x[1])  # primary and secondary label on input 2
    # check if cell of the table is empty (nan is of type "float", otherwise the cell is filled with a list)
    if type(prim_table.loc[prim_label_2][prim_label_1]) is float:
        # if table cell is of type float, it's empty and can be filled with the new values:
        prim_table.loc[prim_label_2][prim_label_1] = [b_star, round(b_star_cf, 3)]
    else:
        # otherwise, we need to check if the degree of certainty of the new sample is higher than the old one. If so, we
        # put the label and degree of certainty of the new sample into the cell
        label, cf = prim_table.loc[prim_label_2][prim_label_1]
        if cf < b_star_cf:
            prim_table.loc[prim_label_2][prim_label_1] = [b_star, round(b_star_cf, 3)]
    # same procedure for the secondary table
    if type(sec_table.loc[sec_label_2][sec_label_1]) is float:
        sec_table.loc[sec_label_2][sec_label_1] = [b_star_star, round(b_star_star_cf, 3)]
    else:
        label, cf = sec_table.loc[sec_label_2][sec_label_1]
        if cf < b_star_star_cf:
            sec_table.loc[sec_label_2][sec_label_1] = [b_star_star, round(b_star_star_cf, 3)]
    return prim_table, sec_table


class FuzzyExample:
    """
    class to create an example with two inputs and one output
    """
    def __init__(self, mu_sigma_list_in_1, mu_sigma_list_in_2, mu_sigma_list_out,
                 labels_in_1, labels_in_2, labels_out,
                 name_in_1, name_in_2, name_out):
        self.fuzzy_in_1 = FuzzyPartition(mu_sigma_list_in_1, labels_in_1, name_in_1)
        self.fuzzy_in_2 = FuzzyPartition(mu_sigma_list_in_2, labels_in_2, name_in_2)
        self.fuzzy_out = FuzzyPartition(mu_sigma_list_out, labels_out, name_out)

    def plot_fuzzy_partitions(self):
        """
        function to plot the gaussian functions (two input, one output) of the fuzzy partitions
        """
        fig, axs = plt.subplots(3, figsize=(15, 15))
        x = np.linspace(0, 1, 100)
        for gaussian, label in zip(self.fuzzy_in_1.fuzzy_partition, self.fuzzy_in_1.labels):
            axs[0].plot(x, gaussian(x), label=label)
        axs[0].set_title(f'Input 1: {self.fuzzy_in_1.name}')
        axs[0].legend(loc='lower left')
        for gaussian, label in zip(self.fuzzy_in_2.fuzzy_partition, self.fuzzy_in_2.labels):
            axs[1].plot(x, gaussian(x), label=label)
        axs[1].set_title(f'Input 2: {self.fuzzy_in_2.name}')
        axs[1].legend(loc='lower left')
        for gaussian, label in zip(self.fuzzy_out.fuzzy_partition, self.fuzzy_out.labels):
            axs[2].plot(x, gaussian(x), label=label)
        axs[2].set_title(f'Output: {self.fuzzy_out.name}')
        axs[2].legend(loc='lower left')
        plt.show()


def predict_sample(sample, alpha, partition_in_1, partition_in_2, partition_out):
    """
    :param sample: sample that should be classified
    :param alpha: alpha value
    :param partition_in_1: FuzzyPartition of input 1
    :param partition_in_2: FuzzyPartition of input 2
    :param partition_out: FuzzyPartition of output
    :return: inferred output, based on the calculation ((24) in the paper)
    """
    b = calc_b(sample, alpha, partition_in_1.fuzzy_partition, partition_in_2.fuzzy_partition)
    # get b_star and b_star_star values of the output, based on the calculation of b
    b_star, b_star_star, b_star_cf, b_star_star_cf = get_b_values(partition_out.fuzzy_partition,
                                                                  partition_out.labels, b)
    # get membership values of the input (sample[0] or sample[1]) with the corresponding fuzzy partition
    membership_values_in_1 = np.array(membership_function(partition_in_1.fuzzy_partition, sample[0]))
    membership_values_in_2 = np.array(membership_function(partition_in_2.fuzzy_partition, sample[1]))
    # get the mu-value of b_star_bar (the gaussian curve with the highest membership value)
    b_star_bar = partition_out.mu_sigma_list[partition_out.labels.index(b_star)][0]
    b_star_star_bar = partition_out.mu_sigma_list[partition_out.labels.index(b_star_star)][0]
    # this lengthy calculation is the formula (24) in the paper
    inferred_output = ((sum(membership_values_in_1 * b_star_bar * b_star_cf
                            + membership_values_in_1 * b_star_star_bar * b_star_star_cf)
                        + sum(membership_values_in_2 * b_star_bar * b_star_cf
                              + membership_values_in_2 * b_star_star_bar * b_star_star_cf))
                       / (sum(membership_values_in_1 * b_star_cf
                              + membership_values_in_1 * b_star_star_cf)
                          + sum(membership_values_in_2 * b_star_cf
                                + membership_values_in_2 * b_star_star_cf)))
    return inferred_output


def main():
    # First, the two input variables:
    driver_style_labels = ['slow', 'average', 'fast']
    driver_style_mu_sigma = [[0.15, 0.15], [0.5, 0.2], [0.85, 0.15]]
    conversation_labels = ['boring', 'ok', 'entertaining']
    conversation_mu_sigma = [[0, 0.3], [0.5, 0.2], [1, 0.3]]
    # Then, the output variable:
    rating_labels = ['terrible', 'bad', 'average', 'good', 'perfect']
    rating_mu_sigma = [[0, 0.1], [0.25, 0.1], [0.5, 0.1], [0.75, 0.1], [1, 0.1]]
    example = FuzzyExample(driver_style_mu_sigma, conversation_mu_sigma, rating_mu_sigma,
                           driver_style_labels, conversation_labels, rating_labels,
                           'Driver style', 'Conversation', 'Rating')
    # plot gaussians of inputs and output
    #example.plot_fuzzy_partitions()
    # create primary and secondary ruletable, filled with nan's
    prim_table, sec_table = create_tables(example.fuzzy_in_1.labels, example.fuzzy_in_2.labels)
    # read in dataset
    train_dat = pd.read_csv('dataset_test.csv', header=0, sep=';')
    # create list of lists, containing the individual samples in the form of [input_1, input_2, output]
    list_of_train_samples = [list(row) for row in train_dat.values]
    alpha = 1 
    # loop over all samples and fill the tables
    for sample in list_of_train_samples:
        prim_table, sec_table = fill_table(example.fuzzy_in_1, example.fuzzy_in_2, example.fuzzy_out,
                                          sample, prim_table, sec_table, alpha)
    # printing the results
    print('--------------------------------------------------------------------------------')
    print('\033[1mPrimary rule table\033[0m'.center(75))
    print(tabulate(prim_table, headers=example.fuzzy_in_1.labels, tablefmt='fancy_grid'))
    print('--------------------------------------------------------------------------------')
    print('\033[1mSecondary rule table\033[0m'.center(75))
    print(tabulate(sec_table, headers=example.fuzzy_in_1.labels, tablefmt='fancy_grid'))

    # classify new_samples
    print('--------------------------------------------------------------------------------')
    print('\033[1mClassifying new samples \033[0m')
    test_dat = pd.read_csv('dataset.csv', header=0, sep=';')
    test_samples = test_dat.iloc[:, :2]
    list_of_test_samples = [list(row) for row in test_samples.values]
    test_y = test_dat.iloc[:, 2]
    inferred_outputs = []
    for sample, ground_truth in zip(list_of_test_samples, test_y):
        inferred_output = predict_sample(sample, alpha, example.fuzzy_in_1, example.fuzzy_in_2, example.fuzzy_out)
        inferred_outputs.append(inferred_output)
        print(f'true model output: {ground_truth}, inferred output: {round(inferred_output, 3)}')

    # assessing performance score of the model
    print('--------------------------------------------------------------------------------')
    print('\033[1mAssessing the performance of our model\033[0m')
    print(f'mean squared error: {round(mean_squared_error(test_y, inferred_outputs), 3)}')

    # testing different alphas
    print('--------------------------------------------------------------------------------')
    print('\033[1mTesting different alphas\033[0m')
    alphas = np.linspace(0.1, 3, 30)
    for a in alphas:
        a = round(a, 1)
        for train_sample in list_of_train_samples:
            prim_table, sec_table = fill_table(example.fuzzy_in_1, example.fuzzy_in_2, example.fuzzy_out,
                                               train_sample, prim_table, sec_table, a)
        inferred_outputs = []
        for test_sample, ground_truth in zip(list_of_test_samples, test_y):
            inferred_output = predict_sample(test_sample, a, example.fuzzy_in_1, example.fuzzy_in_2, example.fuzzy_out)
            inferred_outputs.append(inferred_output)
        print('--------------------------------------------------------------------------------')
        print('--------------------------------------------------------------------------------')
        print(f'\033[1mTraining run for alpha = {a}\033[0m'.center(75))
        print('\033[1mPrimary rule table\033[0m'.center(75))
        print(tabulate(prim_table, headers=example.fuzzy_in_1.labels, tablefmt='fancy_grid'))
        print('--------------------------------------------------------------------------------')
        print('\033[1mSecondary rule table\033[0m'.center(75))
        print(tabulate(sec_table, headers=example.fuzzy_in_1.labels, tablefmt='fancy_grid'))
        print('--------------------------------------------------------------------------------')
        print(f'\033[1mMean squared error for alpha = {a} is '
              f'{round(mean_squared_error(test_y, inferred_outputs), 3)}\033[0m'.center(75))


if __name__ == "__main__":
    main()
