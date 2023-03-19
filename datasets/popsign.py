"""popsign dataset."""

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
from popsign_preprocessing import Preprocessing
from urls import TRAIN_LANDMARK_FILES_URL


# Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
A large-scale dataset that contains isolated american sign language videos.
It contains 250 signs across 94477 videos in total.
"""


class PopSign(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for popsign dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    LANDMARKS_PER_SAMPLE = 543
    INFO = pd.read_csv("train.csv", index_col=0).reset_index()

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        # Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'pose': tfds.features.Tensor(
                    shape=(None, self.LANDMARKS_PER_SAMPLE, 3),
                    dtype=np.float32),
                'label': tfds.features.ClassLabel(
                    names=list(self.INFO["sign"].unique().astype(str)))
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('pose', 'label'),  # Set to `None` to disable
            homepage='https://www.kaggle.com/competitions/asl-signs/data'
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        extracted_path = dl_manager.download_and_extract(
            TRAIN_LANDMARK_FILES_URL)

        # Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(extracted_path, 'train'),
            "validation": self._generate_examples(extracted_path, 'val')
        }

    def _generate_examples(self, source_path, split, cv_split=None):
        """Generator of examples for each split."""

        preprocessing_layer = Preprocessing([])

        if cv_split is None:
            examples = self.get_split(split)
        else:
            examples = self.get_cv_split(split, cv_split)

        for filename, label in examples:
            filename = "/".join(filename.split("/")[1:])
            filepath = source_path / filename
            data = self.load_relevant_data_subset(filepath)
            data = preprocessing_layer.fill_nan_values(data)

            # Yields (key, example)
            yield filename, {
                'pose': data.numpy(),
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
        filenames = self.INFO["path"]
        labels = self.INFO["sign"]
        splits = list(sss.split(filenames, labels))
        train_indices, test_indices = splits[0]
        if split == "train":
            filenames = self.INFO.loc[train_indices, "path"]
            labels = self.INFO.loc[train_indices, "sign"]
        else:
            filenames = self.INFO.loc[test_indices, "path"]
            labels = self.INFO.loc[test_indices, "sign"]
        return zip(filenames, labels)

    def get_cv_split(self, split, cv_split):
        '''
        Obtain a list of filepaths that belong to cv_split=[1, 2, 3, 4, 5]
        of the 'train' or 'val' split
        '''
        skf = StratifiedKFold(n_splits=5,
                              random_state=0,
                              shuffle=True)
        filenames = self.INFO["path"]
        labels = self.INFO["sign"]
        splits = list(skf.split(filenames, labels))
        train_indices, test_indices = splits[cv_split-1]
        if split == "train":
            filenames = self.INFO.loc[train_indices, "path"]
            labels = self.INFO.loc[train_indices, "sign"]
        else:
            filenames = self.INFO.loc[test_indices, "path"]
            labels = self.INFO.loc[test_indices, "sign"]
        return zip(filenames, labels)
