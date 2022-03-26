from .constants import *
import pandas as pd
from .binning.binning_utils import get_bins_map


def process_data(df, num_columns=None, cat_columns=None,
                 num_na_value=NUM_NA_VALUE, cat_na_value=CAT_NA_VALUE, cat_empty_value=CAT_EMPTY_VALUE):
    """
    Returns processed copy of a Pandas DataFrame

    :param df: initial Pandas DataFrame object
    :param num_columns: iterable object of numerical columns names
    :param cat_columns: iterable object of categorical columns names
    :param num_na_value: replacement for numeric NaN values
    :param cat_na_value: replacement for categorical NaN values
    :param cat_empty_value: replacement for categorical empty values
    :return: processed Pandas DataFrame
    """

    df = df.copy()

    if not num_columns:
        num_columns = df.columns[df.dtypes != 'object']
    if not cat_columns:
        cat_columns = df.columns[df.dtypes == 'object']

    for col in num_columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
        df[col].fillna(num_na_value, inplace=True)
    for col in cat_columns:
        df[col].fillna(cat_na_value, inplace=True)
        df[col] = df[col].astype(str).str.lower()
        df[col].replace('', cat_empty_value, inplace=True)

    return df


def bins_to_dummies(X_bin, X_train, y_train, num_features, cat_features):
    """
    Converts binarized features into dummies

    :param X_bin: Pandas Dataframe of binarized features
    :param X_train: Pandas Dataframe of binarized train data
    :param y_train: Pandas Series of train data target
    :param num_features: list of numerical features names
    :param cat_features: list of categorical features names
    :return: bins dummies DataFrame
    """

    features = X_bin.columns
    assert (set(features) == set(X_train.columns))
    assert(set(features) == set(num_features + cat_features))

    binary_df = pd.DataFrame(index=X_bin)
    bins_maps = []

    for feature in features:
        x_train = X_train[feature]
        target_rate = y_train.groupby(x_train).mean()
        share = x_train.value_counts(normalize=True)
        bins_map = get_bins_map(
            target_rate.index,
            target_rate.values,
            [share[b] for b in target_rate.index]
        )
        bins_maps.append(bins_map)

        if feature in num_features:
            bins = sorted(bins_map.keys())
        else:
            bins = list(bins_map.keys())

        ranks = [i for i in len(bins)]
        ranks_dct = dict(zip(bins, ranks))
        x_rank = X_bin[feature].replace(ranks_dct)
        for rank in ranks:
            binary_df[feature + '_' + str(rank)] = (x_rank == rank).astype(int)

    return binary_df, bins_maps
