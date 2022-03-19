from .bin_classes import CategoricalBin
from risk_modeling_tools.modeling.metrics import iv_table
from scipy.optimize import minimize_scalar
import numpy as np


def initialize_categorical_bins(series):
    """
    Initializes CategoricalBin objects in Pandas Series
    cache is used to prevent duplicates
    :param series: Pandas Series object
    :return: Pandas Series with initial bins
    """

    bins_cache = {}

    def get_bin(x):
        b = CategoricalBin()
        b.merge(x)
        cached_b = bins_cache.get(str(b))
        if cached_b:
            return cached_b
        else:
            bins_cache[str(b)] = b
            return b

    feature_bin = series.apply(get_bin)
    return feature_bin


def get_bins_map(bins, share, target_rate):
    """
    Initializes CategoricalBin objects in Pandas Series
    cache is used to prevent duplicates
    :param bins: iterable bins object
    :param share: dict-like share of bins
    :param target_rate: dict-like target rate of bins
    :return: dict of bin's share and target rate
    """

    bins_map = {}
    for b in bins:
        bins_map[b] = {
            'share': share[b],
            'target_rate': target_rate[b]
        }
    return bins_map


def map_bin(x, bins, na_bin):
    """
    Initializes CategoricalBin objects in Pandas Series
    cache is used to prevent duplicates

    :param x: feature value
    :param bins: bins object
    :param na_bin: bin object to fill NA values
    :return: dict of bin's share and target rate
    """

    for b in bins:
        if x in b:
            return b
    return na_bin


def iv_by_threshold(x, y, threshold):
    """
    Transforms feature into binary by given threshold and calculates it's IV

    :param x: Pandas Series feature
    :param y: Pandas Series binary target
    :param threshold: bin object to fill NA values
    :return: IV value of transformed feature
    """

    x = x.copy()
    x = x.apply(lambda a: 0 if a < threshold else 1)
    iv_df = iv_table(x, y)
    iv_value = iv_df.iloc[0]['iv']
    return iv_value


def get_optimal_threshold(x, y, left_bound, right_bound):
    """
    Maximizes numerical feature's IV value for the given bounds

    :param x: Pandas Series numerical feature
    :param y: Pandas Series binary target
    :param left_bound: left bound of the feature values
    :param right_bound: right bound of the feature values
    :return: optimal threshold for maximizing IV
    """

    # assign the main optimization function
    def fun(tr):
        iv_negative = -iv_by_threshold(x, y, tr)
        return iv_negative

    # assign the optimization function for infinite IV case
    def fun_sub(tr):
        left_mask = x <= tr
        right_mask = x > tr
        left_y_rate = y[left_mask].mean()
        right_y_rate = y[right_mask].mean()
        abs_diff_negative = -abs(left_y_rate - right_y_rate)
        return abs_diff_negative

    # if target rate in bounds equals 0 or 1 than optimization cannot be performed
    bounds_mask = (x > left_bound) & (x < right_bound)
    y_rate_bounds = y[bounds_mask].mean()
    if left_bound == right_bound or y_rate_bounds in (0, 1):
        return None
    else:
        opt_result = minimize_scalar(fun, bounds=(left_bound, right_bound), method='bounded')
        optimal_function_value = opt_result['fun']
        if optimal_function_value == -np.inf:
            opt_result = minimize_scalar(fun_sub, bounds=(left_bound, right_bound), method='bounded')
        optimal_threshold = opt_result['x']
        return optimal_threshold
