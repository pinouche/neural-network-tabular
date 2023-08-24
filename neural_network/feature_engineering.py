import numpy as np
import pandas as pd


def get_training_data(x_train, x_val, y_train, y_val):

    k = 100
    corr, top_correlated_features = get_most_correlated_features(x_train, y_train, k)

    x_train = feature_engineering(x_train, top_correlated_features)
    x_val = feature_engineering(x_val, top_correlated_features)

    return x_train, x_val, y_train, y_val


def compute_rank_percentile(df):
    features = [col for col in df.columns if col not in ["date", "id"]]

    def lambda_rank_percentile(d):
        n = d.shape[0] - 1
        d = np.argsort(np.argsort(d, axis=0), axis=0) / n

        return d

    data = df.groupby("date").apply(lambda x: lambda_rank_percentile(x[features]))
    data = np.concatenate(data.values)

    return data


def get_most_correlated_features(x, y, top_k=None):
    features = [col for col in x.columns if col not in ["date", "id"]]

    corr_list = []
    for col in features:
        array_feature = x[col]
        corr = np.corrcoef(array_feature, y["y"])[0, 1]
        corr_list.append(corr)

    if top_k is None:
        top_k = len(features)

    corr_array = np.abs(corr_list)[np.flip(np.argsort(np.abs(corr_list)))]
    top_correlated_features = np.flip(np.argsort(np.abs(corr_list)))[:top_k]
    top_correlated_features = [str(val) for val in top_correlated_features]

    return corr_array, top_correlated_features


def compute_moment(x_data, agg_method="skew", quantile=None, higher_moment=None):
    f = [col for col in x_data.columns if col not in ["date", "id"]]

    if agg_method == "quantile":
        moment_data = x_data.groupby("date").apply(lambda x: np.quantile(x[f], quantile, axis=0)).apply(pd.Series)
    elif agg_method == "count":
        moment_data = x_data.groupby("date").apply(lambda x: x.shape[0]).apply(pd.Series)

    moment_data = np.repeat(moment_data.values, np.unique(x_data["date"], return_counts=True)[1], axis=0)

    return moment_data


def aggregate_quantiles(data):
    new_data = []

    for q in [0.5]:
        agg_df = compute_moment(data, "quantile", q)

        new_data.append(agg_df)

    new_data = np.array(new_data)

    new_data = new_data.reshape((new_data.shape[1], -1))

    return new_data


def feature_engineering(data, top_correlated_features):
    x_most_correlated = pd.concat((data["date"], data[top_correlated_features]), axis=1)

    quantile_data = aggregate_quantiles(x_most_correlated)
    rank_quantile_data = compute_rank_percentile(x_most_correlated)
    count_data = compute_moment(x_most_correlated, agg_method="count")

    data = pd.concat((data.reset_index(drop=True),
                      pd.DataFrame(quantile_data),
                      pd.DataFrame(count_data),
                      pd.DataFrame(rank_quantile_data)), axis=1)

    return data


