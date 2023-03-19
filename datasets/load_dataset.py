import tensorflow_datasets as tfds
import popsign

ds, info = tfds.load('pop_sign', data_dir=".", with_info=True)