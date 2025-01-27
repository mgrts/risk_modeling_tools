from .binning_utils import *
from .bin_classes import NumericalBin
from risk_modeling_tools.constants import *


def reduce_categorical_bins(x, y, max_bins=MAX_BINS):
    x = x.copy()
    n_bins = x.nunique()
    while n_bins > n_bins:
        y_rate = y.groupby(x, sort=False).mean().sort_values()
        y_rate_diff = y_rate.diff()
        min_diff_idx = y_rate_diff.argmin()
        prev_bin = y_rate_diff.index[min_diff_idx - 1]
        curr_bin = y_rate_diff.index[min_diff_idx]
        prev_bin.merge(curr_bin)
        x = x.apply(lambda el: prev_bin if el is curr_bin else el)
        n_bins = x.nunique()
    return x


def binarize_categorical_feature(x, y, min_share=MIN_SHARE, min_diff=MIN_DIFF, max_bins=MAX_BINS):
    """
    Binning algorithm for categorical feature

    :param x: Pandas Series feature
    :param y: Pandas Series target
    :param min_share: minimum share of a bin to stay unmerged
    :param min_diff: minimum difference in target rate between neighbor bins to stay unmerged
    :return: binarized feature, bins map
    """

    x = initialize_categorical_bins(x)
    y_rate = y.groupby(x, sort=False).mean().sort_values()
    share = x.value_counts(normalize=True)
    x_bins = y_rate.keys()
    idx = 1
    # go until reach the last element
    while idx < len(x_bins) - 1:
        prev_bin, curr_bin, next_bin = x_bins[idx - 1], x_bins[idx], x_bins[idx + 1]
        prev_y, curr_y, next_y = y_rate[prev_bin], y_rate[curr_bin], y_rate[next_bin]
        prev_share = share[prev_bin]
        curr_share = share[curr_bin]
        # if previous bin's share is too small
        if prev_share < min_share:
            prev_bin.merge(curr_bin)
            x = x.apply(lambda el: prev_bin if el is curr_bin else el)
        # check target rate and share conditions
        elif abs(prev_y - curr_y) < min_diff or abs(curr_y - next_y) < min_diff or curr_share < min_share:
            # here we calculate which bin is closer
            if abs(prev_y - curr_y) < abs(curr_y - next_y):
                prev_bin.merge(curr_bin)
                x = x.apply(lambda el: prev_bin if el is curr_bin else el)
            else:
                curr_bin.merge(next_bin)
                x = x.apply(lambda el: curr_bin if el is next_bin else el)
        else:
            idx += 1
        # reassign target rate, share and bins
        y_rate = y.groupby(x, sort=False).mean().sort_values()
        share = x.value_counts(normalize=True)
        x_bins = y_rate.keys()

    if len(x_bins) > 1:
        # all elements before -2nd satisfied both target rate and share conditions
        # that means we only need to check last 2 elements
        prev_bin, curr_bin = x_bins[-2], x_bins[-1]
        prev_y, curr_y = y_rate[prev_bin], y_rate[curr_bin]
        prev_share, curr_share = share[prev_bin], share[curr_bin]
        if prev_share < min_share or curr_share < min_share or abs(prev_y - curr_y) < min_diff:
            prev_bin.merge(curr_bin)
            x = x.apply(lambda el: prev_bin if el is curr_bin else el)

    x = reduce_categorical_bins(x, y, max_bins=max_bins)

    return x


def reduce_numerical_bins(x, y, max_bins=MAX_BINS):
    x = x.copy()
    n_bins = x.nunique()
    while n_bins > n_bins:
        y_rate = y.groupby(x).mean()
        y_rate_diff = y_rate.diff().abs()
        min_diff_idx = y_rate_diff.argmin()
        prev_bin = y_rate_diff.index[min_diff_idx - 1]
        curr_bin = y_rate_diff.index[min_diff_idx]
        new_bin = NumericalBin(prev_bin.left, curr_bin.right)
        x = x.apply(lambda a: new_bin if a in (prev_bin, curr_bin) else a)
        n_bins = x.nunique()
    return x


def binarize_numerical_feature(x, y, min_share=MIN_SHARE, min_diff=MIN_DIFF, max_bins=MAX_BINS):
    """
    Binning algorithm for numerical feature

    :param x: Pandas Series feature
    :param y: Pandas Series target
    :param min_share: minimum share of a bin to stay unmerged
    :param min_diff: minimum difference in target rate between neighbor bins to stay unmerged
    :return: binarized feature, bins map
    """
    # initial bounds for optimization
    # border bins should not be less than min_share
    left_bound = x.quantile(min_share)
    right_bound = x.quantile(1 - min_share)
    tr_opt = get_optimal_threshold(x, y, left_bound, right_bound)
    if tr_opt is not None:
        x_left = NumericalBin(-np.inf, tr_opt)
        x_right = NumericalBin(tr_opt, np.inf)
        left_mask = x.apply(lambda a: a in x_left)
        right_mask = x.apply(lambda a: a in x_right)
        y_rate_left = y[left_mask].mean()
        y_rate_right = y[right_mask].mean()
        # check if the initial optimization result follows the target rate condition
        if abs(y_rate_left - y_rate_right) < min_diff:
            return -1
    else:
        return -1

    x_binarized = x.apply(lambda a: x_left if a in x_left else x_right)
    x_bins = sorted(x_binarized.unique())
    x_n = len(x_bins)
    idx = 0
    while idx < x_n:
        curr_interval = x_bins[idx]
        interval_mask = x.apply(lambda a: a in curr_interval)
        x_interval = x[interval_mask]
        # check if current interval can be split
        min_share_to_interval = len(x) * min_share / len(x_interval)
        if min_share_to_interval >= 0.5:
            idx += 1
        else:
            left_bound = x_interval.quantile(min_share_to_interval)
            right_bound = x_interval.quantile(1 - min_share_to_interval)
            tr_opt = get_optimal_threshold(x, y, left_bound, right_bound)
            if tr_opt is not None:
                x_left = NumericalBin(curr_interval.left, tr_opt)
                x_right = NumericalBin(tr_opt, curr_interval.right)
                left_mask = x.apply(lambda a: a in x_left)
                right_mask = x.apply(lambda a: a in x_right)
                y_rate_left = y[left_mask].mean()
                y_rate_right = y[right_mask].mean()
                # check if the optimization result follows the target rate condition
                if abs(y_rate_left - y_rate_right) < min_diff:
                    idx += 1
                else:
                    x_binarized[interval_mask] = x[interval_mask].apply(lambda a: x_left if a in x_left else x_right)
                    x_bins = sorted(x_binarized.unique())
                    x_n = len(x_bins)
            else:
                idx += 1

    # check the minimum target rate difference constraint
    y_rate = y.groupby(x_binarized).mean()
    y_rate_diff = (y_rate - y_rate.shift(-1)).abs()
    min_y_rate_diff_idx = y_rate_diff.argmin()
    min_y_rate_diff = y_rate_diff.iloc[min_y_rate_diff_idx]
    while min_y_rate_diff < min_diff:
        min_y_rate_bin = y_rate_diff.keys()[min_y_rate_diff_idx]
        min_y_rate_next_bin = y_rate_diff.keys()[min_y_rate_diff_idx + 1]
        new_bin = NumericalBin(min_y_rate_bin.left, min_y_rate_next_bin.right)
        x_binarized = x_binarized.apply(lambda a: new_bin if a in (min_y_rate_bin, min_y_rate_next_bin) else a)

        y_rate = y.groupby(x_binarized).mean()
        y_rate_diff = (y_rate - y_rate.shift(-1)).abs()
        min_y_rate_diff_idx = y_rate_diff.argmin()
        min_y_rate_diff = y_rate_diff.iloc[min_y_rate_diff_idx]

    x_binarized = reduce_numerical_bins(x_binarized, y, max_bins=max_bins)

    return x_binarized


def transform_feature(x, bins_map, x_type=None, na_bin=None, cat_na_value=CAT_NA_VALUE, num_na_value=NUM_NA_VALUE):
    """
    Feature transforming algorithm

    :param x: Pandas Series feature
    :param bins_map: fitted bins statistics
    :param x_type: feature's type
    :param na_bin: bin object to fill NA values
    :param cat_na_value: categorical NA value to assign NA bin
    :param num_na_value: numerical NA value to assign NA bin
    :return: binarized feature, bins map
    """

    # assign NA value
    assert x_type in (None, 'categorical', 'numerical')
    x_type = x_type if x_type else 'categorical' if x.dtype == 'object' else 'numerical'
    na_value = cat_na_value if x_type == 'categorical' else num_na_value

    # assign bin for NA or unknown values
    if not na_bin:
        for b in bins_map:
            if na_value in b:
                na_bin = b
                break
        if not na_bin:
            # if NA bin not found assign it to the worst bin
            max_y_bin = max(bins_map, key=lambda a: bins_map[a]['target_rate'])
            na_bin = max_y_bin

    bins = [b for b in bins_map]
    x_transformed = x.apply(lambda a: map_bin(a, bins, na_bin))
    return x_transformed


class Binning:
    """
    A class of a main binning algorithm

    ...

    Attributes
    ----------
    num_features: list
        List of numerical features
    cat_features: list
        List of categorical features
    bins_maps: dict
        Dictionary of bins maps
    min_share: float
        Minimal share of a bin to stay unmerged
    min_diff: float
        self. = False
    fitted : bool
        Flag of the fitting status

    Methods
    -------
    __init__(self, min_share=MIN_SHARE, min_diff=MIN_DIFF)
        Initialize self
    fit(X, y, num_features, cat_features)
        Fits the Binning object to train data
    transform(X, na_bin, cat_na_value, num_na_value)
        Transforms the input data according to train data
    fit_transform(X, na_bin, cat_na_value, num_na_value)
        Sequentially performs fit and transform methods
    """

    def __init__(self, min_share=MIN_SHARE, min_diff=MIN_DIFF, max_bins=MAX_BINS):
        self.num_features = None
        self.cat_features = None
        self.bins_maps = {}
        self.min_share = min_share
        self.min_diff = min_diff
        self.max_bins = max_bins
        self.fitted = False

    def fit(self, X, y, num_features=None, cat_features=None):
        """
        Fit the model according to the given training data.

        :param X: Pandas DataFrame of given training features
        :param y: Pandas Series target
        :param num_features: list of numerical features
        :param cat_features: list of categorical features
        :return: fitted binning
        """
        X = X.copy()
        y = y.copy()
        dtypes = X.dtypes
        if num_features:
            self.num_features = num_features
        else:
            self.num_features = dtypes[dtypes != 'object'].index.to_list()
        if cat_features:
            self.cat_features = cat_features
        else:
            self.cat_features = dtypes[dtypes == 'object'].index.to_list()

        features = X.columns
        for feature in features:
            if feature in self.cat_features:
                x_bin = binarize_categorical_feature(X[feature], y,
                                                     min_share=self.min_share,
                                                     min_diff=self.min_diff,
                                                     max_bins=self.max_bins)
                if type(x_bin) == int:
                    continue
                target_rate = y.groupby(x_bin, sort=False).mean().sort_values()
            elif feature in self.num_features:
                x_bin = binarize_numerical_feature(X[feature], y,
                                                   min_share=self.min_share,
                                                   min_diff=self.min_diff,
                                                   max_bins=self.max_bins)
                if type(x_bin) == int:
                    continue
                target_rate = y.groupby(x_bin).mean()

            share = x_bin.value_counts(normalize=True)
            bins = target_rate.keys()
            bins_map = get_bins_map(bins, share, target_rate)
            self.bins_maps[feature] = bins_map

        self.fitted = True

    def transform(self, X, na_bin=None, cat_na_value=CAT_NA_VALUE, num_na_value=NUM_NA_VALUE):
        """
        Perform binarization of the given data

        :param X: DataFrame of features
        :param na_bin: bin object to fill NA values
        :param cat_na_value: categorical NA value to assign NA bin
        :param num_na_value: numerical NA value to assign NA bin
        :return: transformed features, unknown columns will be dropped
        """
        X = X.copy()
        if self.fitted:
            features = X.columns
            for feature in features:
                if feature in self.cat_features:
                    x_type = 'categorical'
                else:
                    x_type = None
                bin_map = self.bins_maps.get(feature)
                if bin_map:
                    x_transformed = transform_feature(X[feature], bin_map, x_type, na_bin, cat_na_value, num_na_value)
                    X[feature] = x_transformed
                else:
                    X.drop(columns=feature, inplace=True)

            return X

    def fit_transform(self, X, y, num_features=None, cat_features=None,
                      na_bin=None, cat_na_value=CAT_NA_VALUE, num_na_value=NUM_NA_VALUE):
        """
        Fit to data, then transform it

        :param X: Pandas DataFrame of given training features
        :param y: Pandas Series target
        :param num_features: list of numerical features
        :param cat_features: list of categorical features
        :param na_bin: bin object to fill NA values
        :param cat_na_value: categorical NA value to assign NA bin
        :param num_na_value: numerical NA value to assign NA bin
        :return: transformed features, unknown columns will be dropped
        """
        self.fit(X, y, num_features=num_features, cat_features=cat_features)
        X_transformed = self.transform(X, na_bin=na_bin, cat_na_value=cat_na_value, num_na_value=num_na_value)
        return X_transformed
