import pandas as pd


from argparse import Namespace


class Config(Namespace):
    def __init__(self, config):
        for key, value in config.items():
            if isinstance(value, (list, tuple)):
                setattr(self, key, [Config(x) if isinstance(x, dict) else x for x in value])
            else:
                setattr(self, key, Config(value) if isinstance(value, dict) else value)


def load_data():
    data_x = pd.read_parquet('../data/X_train.parquet')
    data_y = pd.read_parquet('../data/y_train.parquet')

    return data_x, data_y


def temporal_train_test_split(data_x, data_y, train_start=0.0, train_end=0.8, val_start=0.8, val_end=1.0):
    #     assert (train_end + 1-val_start) <= 1.0
    assert train_start < train_end
    assert val_start <= val_end

    unique_dates = data_x.date.unique()

    start_date_train = unique_dates[int((len(unique_dates) - 1) * train_start)]
    end_date_train = unique_dates[int((len(unique_dates) - 1) * train_end)]
    x_train = data_x[(data_x['date'] >= start_date_train) & (data_x['date'] <= end_date_train)]
    y_train = data_y[(data_y['date'] >= start_date_train) & (data_y['date'] <= end_date_train)]

    x_val = None
    y_val = None
    if val_start < 1.0:

        split_date_val = unique_dates[int((len(unique_dates) - 1) * val_start)]
        end_date_val = unique_dates[int((len(unique_dates) - 1) * val_end)]

        x_val = data_x[(data_x['date'] > split_date_val) & (data_x['date'] <= end_date_val)]
        y_val = data_y[(data_y['date'] > split_date_val) & (data_y['date'] <= end_date_val)]

        assert len(set(y_train["date"].values).intersection(set(y_val["date"].values))) == 0

    return x_train, x_val, y_train, y_val
