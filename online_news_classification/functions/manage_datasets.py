import os
from operator import index

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def read_json_dataset(filename):
    file = os.path.join(os.getcwd(), os.getenv("DATASETS_FOLDER") + filename)
    df = pd.read_json(file, lines=True)
    return df


def read_csv_dataset(filename, separator):
    file = os.path.join(os.getcwd(), os.getenv("DATASETS_FOLDER") + filename)
    df = pd.read_csv(file, sep=separator)
    return df


def read_csv_dataset_no_dataset(filename, separator):
    file = os.path.join(os.getcwd(), filename)
    df = pd.read_csv(file, sep=separator)
    return df


def read_csv_dataset_with_empty(filename, separator):
    file = os.path.join(os.getcwd(), os.getenv("DATASETS_FOLDER") + filename)
    df = pd.read_csv(file, sep=separator, keep_default_na=False)
    return df


def read_table_dataset(filename, columns):
    file = os.path.join(os.getcwd(), os.getenv("DATASETS_FOLDER") + filename)
    df = pd.read_table(file, header=None, names=columns)
    return df


def save_dataset(dataset, filename):
    file = os.path.join(os.getcwd(), os.getenv("DATASETS_FOLDER") + filename)
    dataset.to_csv(file, sep=";", index=False)


def load_dataset(filename):
    file = os.path.join(os.getenv("DATASETS_FOLDER"), filename)
    return pd.read_csv(file, sep=";", index_col=0)
