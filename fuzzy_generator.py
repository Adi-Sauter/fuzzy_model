import numpy as np
import pandas as pd


# section 3
class Fuzzy:
    def __init__(self):
        pass

    def calc_b(self, x, a, mu):
        """
        :param x: m input-output pairs (x_p,y_p), p=1,2,...,m
        :param a: positive constant
        :param mu: n membership functions
        :return:
        """
        x_p = [sublist[:-1] for sublist in x]
        # x_p = [x for sublist in x_p for x in sublist]  # flatten x_p to get one list without any sublists
        y_p = [sublist[-1] for sublist in x]
        weights = self.calc_compatibility_degree(x_p, mu) ** a
        x_p = np.array(x_p)
        y_p = np.array(y_p)
        weights = np.array(weights)
        b = np.inner(weights, y_p) / sum(weights)
        return b

    def calc_compatibility_degree(self, x, mu):
        """
        :param x: m input-output pairs (x_p,y_p), p=1,2,...,m
        :param mu: n membership functions
        :return: degree of compatibility of input vector x to the fuzzy if-then-rule
        """
        membership_value = mu(x)
        comp_deg = 1
        for val in membership_value:
            comp_deg *= val
        return comp_deg


class RuleTables:
    def create_tables(self, x1, x2):    # index the ruletables by prim_table.loc[x1_label][x2_label]
        """
        :param x1: labels of input 1
        :param x2: labels of input 2
        :return: primary ruletable, secondary ruletable, filled with NaNs, respectively
        """
        prim_table = pd.DataFrame(columns=x1, index=x2)
        sec_table = pd.DataFrame(columns=x1, index=x2)
        return prim_table, sec_table
