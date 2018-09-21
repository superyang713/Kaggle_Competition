import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import StratifiedShuffleSplit


def load_data(filename):
    """
    Load the data into a DataFrame. The data file is located in ./data/
    """
    directory = os.path.abspath(os.path.dirname(__file__))
    filepath = os.path.join(directory, 'data', filename)
    data = pd.read_csv(filepath)
    return data


def save_fig(fig_id, tight_layout=True, fig_extension='png', resolution=300):
    directory = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'image'
    )
    if not os.path.isdir(directory):
        os.mkdir(directory)
    fig_path = os.path.join(directory, fig_id + '.' + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(fig_path, format=fig_extension, dpi=resolution)


def train_test_split_strat(data, col_name, test_size=0.2):
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=42
    )
    for train_index, test_index in split.split(data, data[col_name]):
        train = data.iloc[train_index]
        test = data.iloc[test_index]

    return train, test

