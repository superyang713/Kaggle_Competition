import pandas as pd
import os


def file_path(filename="train.csv"):
    file = os.path.join(os.path.dirname(__file__), filename)
    return file


def load_data(file):
    data = pd.read_csv(file)
    return data
