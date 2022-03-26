import pandas as pd


def optimal_cutoff(x, metric, q, threshold):
    """
    Optimal cutoff search for monotonic target

    :param x: Pandas Series numerical featue
    :param metric: optimization function
    :param q: max number of bins
    :param threshold: max accepted metric value
    :return: bins dummies DataFrame
    """

    x_qcut = pd.qcut(x, q, duplicates='drop')
    x_bins = x_qcut.unique()

    while len(x_bins) > 2:
        x_bin = x_bins[0]
        metric_value = metric(x_bin)

        if metric_value <= threshold:
            x_bins = x_bins[1:]
        else:
            bin_next = x_bins[1]
            bin_next = pd.Interval(x_bin.left, bin_next.right)
            x_bins[1] = bin_next
            x_bins = x_bins[1:]

    opt_cutoff = x_bins[0].left
    return opt_cutoff


def optimal_cutoff_2d(x, y, metric, threshold):
    pass


def tree_based_rule(X, metric, threshold):
    pass
