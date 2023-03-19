import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

info_df = pd.read_csv("/kaggle/input/asl-signs/train.csv", index_col=0)
info_df = info_df.reset_index()


ROWS_PER_FRAME = 543  # number of landmarks per frame


def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


def get_split(info_df, split):
    sss = StratifiedShuffleSplit(n_splits=1,
                                 test_size=0.2,
                                 random_state=0)
    filenames = info_df["path"]
    labels = info_df["sign"]
    splits = list(sss.split(filenames, labels))
    train_indices, test_indices = splits[0]
    if split == "train":
        filenames = info_df.loc[train_indices, "path"]
        labels = info_df.loc[train_indices, "sign"]
    else:
        filenames = info_df.loc[test_indices, "path"]
        labels = info_df.loc[test_indices, "sign"]
    return zip(filenames, labels)


def get_cv_split(info_df, split, cv_split):
    skf = StratifiedKFold(n_splits=5,
                          random_state=0,
                          shuffle=True)
    filenames = info_df["path"]
    labels = info_df["sign"]
    splits = list(skf.split(filenames, labels))
    train_indices, test_indices = splits[cv_split-1]
    if split == "train":
        filenames = info_df.loc[train_indices, "path"]
        labels = info_df.loc[train_indices, "sign"]
    else:
        filenames = info_df.loc[test_indices, "path"]
        labels = info_df.loc[test_indices, "sign"]
    return zip(filenames, labels)


def generate_examples(source_path, split, cv_split=None):
    """Generator of examples for each split."""

    if cv_split is None:
        examples = get_split(info_df, split)
    else:
        examples = get_cv_split(info_df, split, cv_split-1)

    for filename, label in examples:
        filepath = source_path / filename

        # Yields (key, example)
        yield filename, {
            'pose': load_relevant_data_subset(filepath),
            'label': label
        }
