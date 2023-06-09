from enum import IntEnum
import tensorflow as tf
from config import MAX_INPUT_HEIGHT, MAX_LENGTH_IN_DATASET, MIN_INPUT_HEIGHT, NUM_CLASSES
from data_augmentation import RandomFlip2D, RandomRotation2D, RandomScale2D, RandomShift2D, RandomSpeed
from preprocessing import AddRoot, OneItemBatch, Center, FillBlueWithAngle, PadIfLessThan, RemoveZ, ResizeIfMoreThan, SortColumns, TranslationScaleInvariant, OneItemUnbatch
import tensorflow_datasets as tfds
from skeleton_graph import tssi_v2

TSSI_ORDER = tssi_v2()[1]


class LayerType(IntEnum):
    Augmentation = 1
    Normalization = 2
    Data = 3


LayerDict = {
    'random_speed': {
        'type': LayerType.Augmentation,
        'layer': RandomSpeed(min_frames=12, max_frames=44, seed=5),
    },
    'random_rotation': {
        'type': LayerType.Augmentation,
        'layer': RandomRotation2D(factor=15.0, min_value=0.0, max_value=1.0, seed=4),
    },
    'random_flip': {
        'type': LayerType.Augmentation,
        'layer': RandomFlip2D("horizontal", min_value=0.0, max_value=1.0, seed=3),
    },
    'random_scale': {
        'type': LayerType.Augmentation,
        'layer': RandomScale2D(min_value=0.0, max_value=1.0, seed=1),
    },
    'random_shift': {
        'type': LayerType.Augmentation,
        'layer': RandomShift2D(min_value=0.0, max_value=1.0, seed=2)
    },
    'invariant_frame': {
        'type': LayerType.Normalization,
        'layer': TranslationScaleInvariant(level="frame")
    },
    'invariant_joint': {
        'type': LayerType.Normalization,
        'layer': TranslationScaleInvariant(level="joint")
    },
    'center': {
        'type': LayerType.Normalization,
        'layer': Center(around_index=0)
    },
    'train_resize': {
        'type': LayerType.Normalization,
        'layer': ResizeIfMoreThan(frames=MIN_INPUT_HEIGHT)
    },
    'test_resize': {
        'type': LayerType.Normalization,
        'layer': ResizeIfMoreThan(frames=MAX_INPUT_HEIGHT)
    },
    'pad': {
        'type': LayerType.Normalization,
        'layer': PadIfLessThan(frames=MIN_INPUT_HEIGHT)
    },
    'angle': {
        'type': LayerType.Data,
        'layer': FillBlueWithAngle(x_channel=0, y_channel=1, scale_to=[0, 1]),
    },
    'norm_imagenet': {
        'type': LayerType.Normalization,
        'layer': tf.keras.layers.Normalization(axis=-1,
                                               mean=[0.485, 0.456, 0.406],
                                               variance=[0.052441, 0.050176, 0.050625]),
    },
    # placeholder for layer, mean and variance are obtained dinamically
    'norm': {
        'type': LayerType.Normalization,
        'layer': tf.keras.layers.Normalization(axis=-1,
                                               mean=[248.08896, 246.56985, 0.],
                                               variance=[9022.948, 17438.518, 0.])
    },
    'remove_z': {
        'type': LayerType.Data,
        'layer': RemoveZ()
    }
}


# Augmentation Order = ['speed', 'rotation', 'flip', 'scale', 'shift']
PipelineDict = {
    'default': {
        'train': ['random_speed', 'random_flip', 'random_scale', 'train_resize', 'pad'],
        'test': ['test_resize', 'pad']
    }
}


@tf.function
def extract_pose(item):
    return item["data"]


COMMON_PREPROCESSING = tf.keras.Sequential([
    # OneItemBatch(),
    # AddRoot(),
    # SortColumns(tssi_order=TSSI_ORDER),
    # OneItemUnbatch()
])


@tf.function
def base_processing(item):
    pose = item["data"]
    label = item["label"]
    pose = COMMON_PREPROCESSING(pose)
    one_hot_label = tf.one_hot(label, NUM_CLASSES)
    return pose, one_hot_label


def generate_train_dataset(dataset,
                           train_map_fn,
                           repeat=False,
                           batch_size=64,
                           buffer_size=5000,
                           deterministic=False):
    # apply base preprocessing
    # add root, sort columns and one-hot label encoding
    # it returns an unbatched dataset
    ds = dataset.map(
        base_processing,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    ).cache()

    # shuffle, map and batch dataset
    if deterministic:
        train_dataset = ds \
            .shuffle(buffer_size) \
            .map(train_map_fn) \
            .batch(batch_size)
    else:
        train_dataset = ds \
            .shuffle(buffer_size) \
            .map(train_map_fn,
                 num_parallel_calls=tf.data.AUTOTUNE,
                 deterministic=False) \
            .batch(batch_size) \
            .prefetch(tf.data.AUTOTUNE)

    if repeat:
        train_dataset = train_dataset.repeat()

    return train_dataset


def generate_test_dataset(dataset,
                          test_map_fn,
                          batch_size=64):
    # apply base preprocessing
    # add root, sort columns and one-hot label encoding
    # it returns an unbatched dataset
    ds = dataset.map(
        base_processing,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )

    # batch dataset per length
    def element_length_func(x, y): return tf.shape(x)[0]
    max_element_length = MAX_LENGTH_IN_DATASET
    bucket_boundaries = list(range(1, max_element_length))
    bucket_batch_sizes = [batch_size] * max_element_length
    ds = ds.bucket_by_sequence_length(
        element_length_func=element_length_func,
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=bucket_batch_sizes,
        no_padding=True)

    # map dataset and cache
    dataset = ds \
        .map(test_map_fn,
             num_parallel_calls=tf.data.AUTOTUNE,
             deterministic=False) \
        .cache()

    return dataset


def build_pipeline(pipeline, exclude_augmentation=False, split="train"):
    # normalization: None, str or list
    if pipeline == None:
        layers = []
    elif type(pipeline) is str:
        items = [LayerDict[name] for name in PipelineDict[pipeline][split]]
        if exclude_augmentation:
            items = [item for item in items if
                     item["type"] != LayerType.Augmentation]
        layers = [item["layer"] for item in items]
    else:
        raise Exception("Pipeline " +
                        str(pipeline) + " not found")
    pipeline = tf.keras.Sequential(layers, name="normalization")
    return pipeline


class Dataset():
    def __init__(self, concat_validation_to_train=False):
        global LayerDict

        # obtain dataset
        ds, info = tfds.load(
            'pop_sign_tssi', data_dir="datasets", with_info=True)

        # generate train dataset
        if concat_validation_to_train:
            ds["train"] = ds["train"].concatenate(ds["validation"])

        # generate norm layer
        # norm = tf.keras.layers.Normalization(axis=-2)
        # norm.adapt(ds["train"].map(extract_pose,
        #                            num_parallel_calls=tf.data.AUTOTUNE))
        # LayerDict["norm"]["layer"] = norm

        # obtain characteristics of the dataset
        num_train_examples = ds["train"].cardinality()
        num_val_examples = ds["validation"].cardinality()
        num_test_examples = 0
        num_total_examples = num_train_examples + num_val_examples + num_test_examples

        self.ds = ds
        self.num_train_examples = num_train_examples
        self.num_val_examples = num_val_examples
        self.num_test_examples = num_test_examples
        self.num_total_examples = num_total_examples
        self.input_width = len(TSSI_ORDER)

    def get_training_set(self,
                         batch_size=64,
                         buffer_size=5000,
                         repeat=False,
                         deterministic=False,
                         augmentation=True,
                         pipeline="default"):
        # define pipeline
        exclude_augmentation = not augmentation
        preprocessing_pipeline = build_pipeline(
            pipeline, exclude_augmentation, "train")

        # define the train map function
        @tf.function
        def train_map_fn(x, y):
            batch = tf.expand_dims(x, axis=0)
            batch = preprocessing_pipeline(batch, training=True)
            x = tf.ensure_shape(
                batch[0], [MIN_INPUT_HEIGHT, len(TSSI_ORDER), 2])
            return x, y

        train_ds = self.ds["train"]

        dataset = generate_train_dataset(train_ds,
                                         train_map_fn,
                                         repeat=repeat,
                                         batch_size=batch_size,
                                         buffer_size=buffer_size,
                                         deterministic=deterministic)

        return dataset

    def get_validation_set(self,
                           batch_size=64,
                           pipeline="default"):
        # define pipeline
        preprocessing_pipeline = build_pipeline(
            pipeline, exclude_augmentation=True, split="test")

        # define the val map function
        @tf.function
        def test_map_fn(batch_x, batch_y):
            batch_x = preprocessing_pipeline(batch_x)
            return batch_x, batch_y

        val_ds = self.ds["validation"]

        dataset = generate_test_dataset(val_ds,
                                        test_map_fn,
                                        batch_size=batch_size)

        return dataset

    def get_testing_set(self,
                        batch_size=64,
                        pipeline="default"):
        # define pipeline
        preprocessing_pipeline = build_pipeline(
            pipeline, exclude_augmentation=True, split="test")

        # define the val map function
        @tf.function
        def test_map_fn(batch_x, batch_y):
            batch_x = preprocessing_pipeline(batch_x)
            return batch_x, batch_y

        test_ds = self.ds["test"]

        dataset = generate_test_dataset(test_ds,
                                        test_map_fn,
                                        batch_size=batch_size)

        return dataset
