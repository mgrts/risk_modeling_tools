from sklearn.linear_model import LogisticRegression
import pandas as pd
from .metrics import iv_table, gini_normalized
from risk_modeling_tools.binning.binning_algorithms import Binning


def paired_correlations(df):
    """
    Calculates paired Pearson's correlations for dataframe

    :param df: dataframe of numerical features
    :return: Pandas DataFrame of paired correlations
    """

    correlations = df.corr().abs().unstack().sort_values().reset_index()
    correlations = correlations[correlations['level_0'] != correlations['level_1']]
    return correlations


def drop_correlated_features(X, y, threshold=0.6):
    """
    Iteratively drops highly correlated features from dataframe
    feature to drop takes by minimal IV

    :param X: Pandas DataFrame of numerical features
    :param y: Pandas Series binary target
    :param threshold: correlation threshold
    :return: Pandas DataFrame of non-correlated features
    """

    no_corr_df = X.copy()
    correlations = paired_correlations(no_corr_df)
    features = (correlations.iloc[-1]['level_0'], correlations.iloc[-1]['level_1'])
    max_corr = correlations.iloc[-1][0]

    while max_corr > threshold:
        iv_0 = iv_table(X[features[0]], y).iloc[0]['iv']
        iv_1 = iv_table(X[features[1]], y).iloc[0]['iv']

        if iv_0 < iv_1:
            no_corr_df.drop(features[0], axis=1, inplace=True)
        else:
            no_corr_df.drop(features[1], axis=1, inplace=True)

        correlations = paired_correlations(no_corr_df)
        features = (correlations.iloc[-1]['level_0'], correlations.iloc[-1]['level_1'])
        max_corr = correlations.iloc[-1][0]

    return no_corr_df


def drop_unstable_features(X, y, date, n_periods=10, threshold=0.9):
    """
    Drops unstable features from dataframe

    :param X: Pandas DataFrame of features
    :param y: Pandas Series binary target
    :param n_periods: number of periods for stability index calculation
    :param threshold: stability index threshold
    :return: Pandas DataFrame of non-correlated features
    """

    stable_features = []
    periods = pd.qcut(date.rank(method='dense'), n_periods, duplicates='drop')
    indexes = [periods[periods == p].index for p in sorted(periods.unique())]
    features = X.columns

    for feature in features:
        sequence = y.groupby(X[feature]).mean().sort_values().index
        numerator = 0
        denominator = len(sequence) * len(indexes)

        for index in indexes:
            index_sequence = y.loc[index].groupby(X[feature].loc[index]).mean().sort_values().keys()

            if len(index_sequence) == len(sequence):
                numerator += sum(sequence == index_sequence)

        if numerator / denominator >= threshold:
            stable_features.append(feature)

    stable_df = X[stable_features].copy()

    return stable_df


def iv_report(X, y):
    """
    Information Value calculation for features

    :param X: Pandas DataFrame of features
    :param y: Pandas Series binary target
    :return: Pandas DataFrame of IV
    """

    binning = Binning()
    bin_df = binning.fit_transform(X, y)

    iv_df = pd.DataFrame(index=X.columns)
    iv_df['iv'] = iv_df.index.apply(lambda x: iv_table(bin_df[x], y)['iv'].iloc[0])
    iv_df.sort_values(by='iv', ascending=False, inplace=True)

    return iv_df


def train_logistic_regression(X_train, y_train, X_val, y_val, min_coef=1, max_iter=2000):
    """
    Logistic regression training algorithm on binary features

    :param X_train: Pandas DataFrame of train set features
    :param y_train: Pandas Series of train set binary target
    :param X_val: Pandas DataFrame of validation set features
    :param y_val: Pandas Series of validation set binary target
    :param min_coef: minimal absolute coefficient value to keep feature in model
    :param max_iter: maximum number of iterations to converge
    :return: Pandas DataFrame of ginies on every iteration, list of feature sets, list of models
    """

    X_train_short = X_train.copy()
    X_val_short = X_val.copy()

    iteration = []
    gini_train = []
    gini_val = []
    feature_sets = []
    models = []

    i = 1
    min_abs_coef_value = 0
    while min_abs_coef_value < min_coef:
        log_reg = LogisticRegression(max_iter=max_iter)
        log_reg.fit(X_train_short, y_train)
        p_train = log_reg.predict_proba(X_train_short)[:, 1]
        p_val = log_reg.predict_proba(X_val_short)[:, 1]
        coef_abs = dict(zip(X_train_short.keys(), abs(log_reg.coef_[0])))
        min_abs_coef_value = min(coef_abs.values())
        min_abs_coef_name = min(coef_abs, key=coef_abs.get)

        iteration.append(i)
        gini_train.append(gini_normalized(y_train, p_train))
        gini_val.append(gini_normalized(y_val, p_val))
        feature_sets.append(coef_abs.keys())
        models.append(log_reg)

        X_train_short.drop(min_abs_coef_name, axis=1, inplace=True)
        X_val_short.drop(min_abs_coef_name, axis=1, inplace=True)
        i += 1

    results_df = pd.DataFrame({
        'iteration': iteration,
        'gini_train': gini_train,
        'gini_val': gini_val
    })

    return results_df, feature_sets, models
