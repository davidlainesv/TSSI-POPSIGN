"""popsigntssi dataset."""

import sys
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
import json

sys.path.insert(0, "../")
from skeleton_graph import tssi_v2
from preprocessing import OneItemBatch, FillNaNValues, RemoveZ, OneItemUnbatch, AddRoot, SortColumns

TSSI_ORDER = tssi_v2()[1]
SOURCE_PATH = Path("./asl-signs")

# Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
A large-scale dataset that contains isolated american sign language videos.
It contains 250 signs across 94477 videos in total.
"""


def get_dataset_list():
    df = pd.read_csv(SOURCE_PATH / "train.csv", index_col=0)
    df = df.reset_index()
    return df


def get_sign_list():
    f = open(SOURCE_PATH / "sign_to_prediction_index_map.json")
    sign_dict = json.load(f)
    sorted_sign_items = sorted(sign_dict.items(), key=lambda x: x[1])
    sign_list = [item[0] for item in sorted_sign_items]
    return sign_list


class PopSignTssi(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for popsignTssi dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    LANDMARKS_PER_SAMPLE = 543
    DATASET_LIST = get_dataset_list()
    SIGN_LIST = get_sign_list()
    DATA_SHAPE = (None, len(TSSI_ORDER), 2)

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        # Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'data': tfds.features.Tensor(shape=self.DATA_SHAPE, dtype=np.float32),
                'label': tfds.features.ClassLabel(names=self.SIGN_LIST)
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('data', 'label'),  # Set to `None` to disable
            homepage='https://www.kaggle.com/competitions/asl-signs/data'
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        # Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(SOURCE_PATH, 'train'),
            "validation": self._generate_examples(SOURCE_PATH, 'val')
        }

    def _generate_examples(self, source_path, split, cv_split=None):
        """Generator of examples for each split."""

        preprocessing_pipeline = tf.keras.Sequential([
            OneItemBatch(),
            RemoveZ(),
            FillNaNValues(),
            AddRoot(),
            SortColumns(tssi_order=TSSI_ORDER),
            OneItemUnbatch()
        ])

        if cv_split is None:
            examples = self.get_split(split)
        else:
            examples = self.get_cv_split(split, cv_split)

        for filename, label in examples:
            filepath = source_path / filename
            data = self.load_relevant_data_subset(filepath)
            data = preprocessing_pipeline(data)

            # Yields (key, example)
            yield filename, {
                'data': data.numpy(),
                'label': label
            }

    def load_relevant_data_subset(self, pq_path):
        '''
        Each video is loaded with the following function
        '''
        data_columns = ['x', 'y', 'z']
        data = pd.read_parquet(pq_path, columns=data_columns)
        n_frames = int(len(data) / self.LANDMARKS_PER_SAMPLE)
        data = data.values.reshape(
            n_frames, self.LANDMARKS_PER_SAMPLE, len(data_columns))
        return data.astype(np.float32)

    def get_split(self, split):
        '''
        Obtain a list of filepaths that belong to 'train' or 'val' split
        '''
        sss = StratifiedShuffleSplit(n_splits=1,
                                     test_size=0.2,
                                     random_state=0)
        filenames = self.DATASET_LIST["path"]
        labels = self.DATASET_LIST["sign"]
        splits = list(sss.split(filenames, labels))
        train_indices, test_indices = splits[0]
        if split == "train":
            filenames = self.DATASET_LIST.loc[train_indices, "path"]
            labels = self.DATASET_LIST.loc[train_indices, "sign"]
        else:
            filenames = self.DATASET_LIST.loc[test_indices, "path"]
            labels = self.DATASET_LIST.loc[test_indices, "sign"]
        return zip(filenames, labels)

    def get_cv_split(self, split, cv_split):
        '''
        Obtain a list of filepaths that belong to cv_split=[1, 2, 3, 4, 5]
        of the 'train' or 'val' split
        '''
        skf = StratifiedKFold(n_splits=5,
                              random_state=0,
                              shuffle=True)
        filenames = self.DATASET_LIST["path"]
        labels = self.DATASET_LIST["sign"]
        splits = list(skf.split(filenames, labels))
        train_indices, test_indices = splits[cv_split-1]
        if split == "train":
            filenames = self.DATASET_LIST.loc[train_indices, "path"]
            labels = self.DATASET_LIST.loc[train_indices, "sign"]
        else:
            filenames = self.DATASET_LIST.loc[test_indices, "path"]
            labels = self.DATASET_LIST.loc[test_indices, "sign"]
        return zip(filenames, labels)
