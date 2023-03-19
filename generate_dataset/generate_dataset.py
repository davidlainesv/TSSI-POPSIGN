import numpy as np
import tensorflow_datasets as tfds
from .utils import generate_examples
from .utils import ROWS_PER_FRAME
from .utils import info_df

from pathlib import Path

extracted_path = Path("/kaggle/input/asl-signs")

# !rm -r "/kaggle/working/datasets"

tfds.dataset_builders.store_as_tfds_dataset(
    name="popsign",
    version=tfds.core.Version('1.0.0'),
    release_notes={'1.0.0': 'Initial release.'},
    data_dir="/kaggle/working/datasets",
    split_datasets={
        "train": generate_examples(extracted_path, 'train'),
        "validation": generate_examples(extracted_path, 'validation'),
    },
    features=tfds.features.FeaturesDict({
        'pose': tfds.features.Tensor(shape=(None, ROWS_PER_FRAME, 3), dtype=np.float32),
        'label': tfds.features.ClassLabel(names=info_df["sign"].unique().tolist())
    })
)
