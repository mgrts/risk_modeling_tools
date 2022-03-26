import numpy as np
import pandas as pd


def gini(actual, predicted):
    """
    Calculates Gini

    :param actual: array-like binary target
    :param predicted: array-like predicted values
    :return: float Gini value
    """

    a = np.asarray(np.c_[actual, predicted, np.arange(len(actual))], dtype=np.float)
    a = a[np.lexsort((a[:, 2], -1 * a[:, 1]))]
    total_losses = a[:, 0].sum()
    gini_sum = a[:, 0].cumsum().sum() / total_losses
    gini_sum -= (len(actual) + 1) / 2.
    return gini_sum / len(actual)


def gini_normalized(actual, predicted):
    """
    Calculates Normalized Gini

    :param actual: array-like binary target
    :param predicted: array-like predicted values
    :return: float normalized gini value in 0-1 range
    """

    return gini(actual, predicted) / gini(actual, actual)


def iv_table(x, y):
    """
    Calculates WoE-IV table

    :param x: array-like feature
    :param y: array-like binary target
    :return: Pandas DataFrame of WoE-IV statistics
    """

    table = pd.crosstab(x, y, normalize='columns') \
        .assign(woe=lambda dfx: np.log(dfx[1] / dfx[0])) \
        .assign(iv=lambda dfx: np.sum(dfx['woe'] * (dfx[1] - dfx[0])))
    return table
