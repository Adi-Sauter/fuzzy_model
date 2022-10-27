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
    :param x: list with output in last position and inputs as other elements
    :param a: positive constant
    :param partition: list of membership functions in the form of gaussian distributions
    :return: b
    """
    x_p = x[:-1]
    #print(f'x_p: {x_p}')
    #x_p = [sublist[:-1] for sublist in x]
    # x_p = [x for sublist in x_p for x in sublist]  # flatten x_p to get one list without any sublists
    y_p = x[-1]
    #print(f'y_p: {y_p}')
    #y_p = [sublist[-1] for sublist in x]
    weights = [calc_compatibility_degree(x_p[0], partition_in_1) ** a,
               calc_compatibility_degree(x_p[1], partition_in_2) ** a]
    #print(f'weights: {weights}')
    #x_p = np.array(x_p)
    y_p = np.array(y_p)
    weights = np.array(weights)
    b = sum(weights*y_p) / sum(weights)
    #print(f'b: {b}')
    return b


def calc_compatibility_degree(x, partition):
    """
    :param x: list of input values
    :param partition: list of membership functions in the form of gaussian distributions
    :return: degree of compatibility of input vector x to the fuzzy if-then-rule
    """
    #print(f'x in calc_compatibility_degree: {x}')
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
    #print(f'x in membership function: {x}')
    return [gaussian(x) for gaussian in partition]


def sort_by_membership_value(partition, x):
    """
    :param partition: list of membership functions in the form of gaussian distributions
    :param x: input value
    :return: list of indices, sorted by decreasing membership value
    """
    #print(f'x in sort_by_membership_value: {x}')
    membership_values = membership_function(partition, x)
    sorted_indices = np.argsort(membership_values)
    sorted_membership_values = np.sort(membership_values)
    return sorted_indices, sorted_membership_values


def get_b_values(partition, labels, b):
    """
    :param partition: list of membership functions in the form of gaussian distributions
    :param labels: names of the fuzzy sets in the partition
    :param b: calculated b-value
    :return:
    """
    #print(f'b in get_b_values: {b}')
    sorted_labels, sorted_values = sort_by_membership_value(partition, b)  # in ascending order
    b_star = labels[sorted_labels[-1]]
    b_star_star = labels[sorted_labels[-2]]
    b_star_cf = sorted_values[-1]
    b_star_star_cf = sorted_values[-2]
    return b_star, b_star_star, b_star_cf, b_star_star_cf


def fill_table(partition_in_1, partition_in_2, partition_out, x, prim_table, sec_table, a):
    """
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
        get_b_values(partition_out.fuzzy_partition, partition_out.labels, b)  # these are the labels of the output
    prim_label_1, sec_label_1, prim_1_cf, sec_1_cf = \
        get_b_values(partition_in_1.fuzzy_partition, partition_in_1.labels, x[0])  # calculate argmax(mu(x))
    print(f'prim_label_1: {prim_label_1}, sec_label_1: {sec_label_1}, prim_1_cf: {prim_1_cf}, sec_1_cf: {sec_1_cf}')
    prim_label_2, sec_label_2, prim_2_cf, sec_2_cf = \
        get_b_values(partition_in_2.fuzzy_partition, partition_in_2.labels, x[1])  # calculate argmax_2(mu(x))
    print(f'prim_label_2: {prim_label_2}, sec_label_2: {sec_label_2}, prim_2_cf: {prim_2_cf}, sec_2_cf: {sec_2_cf}')
    #if np.isnan(prim_table.loc[prim_label_1][prim_label_2]).any():
    if type(prim_table.loc[prim_label_2][prim_label_1]) is float:
        prim_table.loc[prim_label_2][prim_label_1] = [b_star, b_star_cf]
    else:
        label, cf = prim_table.loc[prim_label_2][prim_label_1]
        if cf < b_star_cf:
            prim_table.loc[prim_label_2][prim_label_1] = [b_star, b_star_cf]
    #if np.isnan(sec_table.loc[sec_label_1][sec_label_2]).any():
    if type(sec_table.loc[sec_label_2][sec_label_1]) is float:
        sec_table.loc[sec_label_2][sec_label_1] = [b_star_star, b_star_star_cf]
    else:
        label, cf = sec_table.loc[sec_label_2][sec_label_1]
        if cf < b_star_star_cf:
            sec_table.loc[sec_label_2][sec_label_1] = [b_star_star, b_star_star_cf]
    return prim_table, sec_table


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
        fig, axs = plt.subplots(3, figsize=(10,10))
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
    #example.plot_fuzzy_partitions()
    prim_table, sec_table = create_tables(example.fuzzy_in_1.labels, example.fuzzy_in_2.labels)
    dat = pd.read_csv('dataset.csv', header=0, sep=';')
    list_of_samples = [list(row) for row in dat.values]
    alpha = 1
    for sample in list_of_samples:
        prim_table, sec_table = fill_table(example.fuzzy_in_1, example.fuzzy_in_2, example.fuzzy_out,
                                           sample, prim_table, sec_table, alpha)


if __name__ == "__main__":
    main()





