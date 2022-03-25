import numpy as np
import pandas as pd


class NumericalBin(pd.Interval):
    """
    A class used to represent a bin of numerical feature

    ...

    Attributes
    ----------
    See pd.Interval documentation

    Methods
    -------
    merge(numerical_bin)
        Merges current bin with a new one
    """

    def merge(self, numerical_bin):
        if type(numerical_bin) is NumericalBin:
            if numerical_bin.right < self.left or numerical_bin.left > self.right:
                raise AssertionError('Empty space between intervals or one of the intervals includes another')
            else:
                if self.left == numerical_bin.right:
                    self.__init__(numerical_bin.left, self.right)
                else:
                    self.__init__(self.left, numerical_bin.right)


class CategoricalBin:
    """
    A class used to represent a bin a of categorical feature

    ...

    Attributes
    ----------
    bin_elements : np.array
        an array of bin elements

    Methods
    -------
    add_element(new_elements)
        Adds new element to the bin
    """

    def __init__(self):
        self.bin_elements = np.array([])

    def __str__(self):
        return str(self.bin_elements)

    def __contains__(self, item):
        return item in self.bin_elements

    def merge(self, new_element):
        if type(new_element) is CategoricalBin:
            self.bin_elements = np.concatenate([self.bin_elements, new_element.bin_elements])
        else:
            self.bin_elements = np.append(self.bin_elements, new_element)
