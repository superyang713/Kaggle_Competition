import pandas as pd
import os


def load_data(filename):
    directory = os.path.abspath(os.path.dirname(__file__))
    filepath = os.path.join(directory, 'data', filename)
    data = pd.read_csv(filepath)
    return data
