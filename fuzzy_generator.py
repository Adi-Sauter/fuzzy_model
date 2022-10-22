import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# section 3


def create_tables(x1, x2):    # index the ruletables by prim_table.loc[x1_label][x2_label]
    """
    :param x1: labels of input 1
    :param x2: labels of input 2
    :return: primary ruletable, secondary ruletable, filled with NaNs, respectively
    """
    prim_table = pd.DataFrame(columns=x1, index=x2)
    sec_table = pd.DataFrame(columns=x1, index=x2)
    return prim_table, sec_table


class FuzzyPartition:
    def __init__(self, mu_sigma_list, labels, name):
        self.fuzzy_partition = self.generate_fuzzy_partition(mu_sigma_list)
        self.labels = labels
        self.name = name

    def create_gaussian(self, mu, sigma):
        """
        :param mu: mean of distribution
        :param sigma: standard deviation of distribution
        :return: function of gaussian distribution with mu and sigma as the relevant values
        """
        return lambda x: np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

    def generate_fuzzy_partition(self, mu_sigma_list):
        """
        :param mu_sigma_list:
        :return: list of gaussians representing the fuzzy set
        """
        gaussians = []
        for sublist in mu_sigma_list:
            mu = sublist[0]
            sigma = sublist[1]
            gaussians.append(self.create_gaussian(mu, sigma))
        return gaussians


def calc_b(x, a, partition):
    """
    :param x: m input-output pairs (x_p,y_p), p=1,2,...,m
    :param a: positive constant
    :param partition: list of membership functions in the form of gaussian distributions
    :return:
    """
    x_p = [sublist[:-1] for sublist in x]
    # x_p = [x for sublist in x_p for x in sublist]  # flatten x_p to get one list without any sublists
    y_p = [sublist[-1] for sublist in x]
    weights = calc_compatibility_degree(x_p, partition) ** a
    #x_p = np.array(x_p)
    y_p = np.array(y_p)
    weights = np.array(weights)
    b = np.inner(weights, y_p) / sum(weights)
    return b


def calc_compatibility_degree(x, partition):
    """
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
    :param partition: list of membership functions in the form of gaussian distributions
    :param x: input value
    :return: list of membership values of x with each of the gaussians
    """
    return [gaussian(x) for gaussian in partition]


def sort_by_membership_value(partition, x):
    """
    :param partition: list of membership functions in the form of gaussian distributions
    :param x: input value
    :return: list of indices, sorted by decreasing membership value
    """
    return np.argsort(membership_function(partition, x))   #, sorted(membership_function(partition, x))


def get_b_values(partition, labels, b):
    """
    :param partition: list of membership functions in the form of gaussian distributions
    :param labels: names of the fuzzy sets in the partition
    :param b: calculated b-value
    :return:
    """
    sorted_values = sort_by_membership_value(partition, b) # in ascending order
    b_star = labels[sorted_values[-1]]
    b_star_star = labels[sorted_values[-2]]
    return b_star, b_star_star


def fill_table(partition, labels, x, prim_table, sec_table, a, input_1_label, input_2_label):
    b = calc_b(x, a, partition)
    b_star, b_star_star = get_b_values(partition, labels, b)  # these are the labels of the output
    prim_table[input_1_label, input_2_label] = [b_star] # add degree of certainty here!!


'''
def get_cf_values(b_star, b_star_star, b_star_val, b_star_star_val):
    """
    :param b_star: fuzzy set with the highest membership value for input
    :param b_star_star: fuzzy set with the second-highest membership value for input
    :param b_star_val: highest membership value for input of one fuzzy set (b_star)
    :param b_star_star_val: second-highest membership value for input of one fuzzy set (b_star_star)
    :return: degrees of certainty for b_star and b_star_star
    """
    pass
'''


class FuzzyExample:
    def __init__(self, mu_sigma_list_in_1, mu_sigma_list_in_2, mu_sigma_list_out,
                 labels_in_1, labels_in_2, labels_out,
                 name_in_1, name_in_2, name_out):
        self.fuzzy_in_1 = FuzzyPartition(mu_sigma_list_in_1, labels_in_1, name_in_1)
        self.fuzzy_in_2 = FuzzyPartition(mu_sigma_list_in_2, labels_in_2, name_in_2)
        self.fuzzy_out = FuzzyPartition(mu_sigma_list_out, labels_out, name_out)

    def plot_fuzzy_partitions(self):
        fig, axs = plt.subplots(3)
        x = np.linspace(0, 5, 100)
        for gaussian in self.fuzzy_in_1.fuzzy_partition:
            axs[0].plot(x, gaussian(x))
        axs[0].set_title(f'Input 1: {self.fuzzy_in_1.name}')
        for gaussian in self.fuzzy_in_2.fuzzy_partition:
            axs[1].plot(x, gaussian(x))
        axs[1].set_title(f'Input 2: {self.fuzzy_in_2.name}')
        for gaussian in self.fuzzy_out.fuzzy_partition:
            axs[2].plot(x, gaussian(x))
        axs[2].set_title(f'Output: {self.fuzzy_out.name}')


def main():
    #example = FuzzyExample()
    #example.plot_fuzzy_sets()
    #prim_table, sec_table = create_tables(example.fuzzy_in_1.labels, example.fuzzy_in_2.labels)
    pass


if __name__ == "__main__":
    main()





