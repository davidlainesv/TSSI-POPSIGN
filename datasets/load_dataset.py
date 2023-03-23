import tensorflow_datasets as tfds
import popsigntssi

ds, info = tfds.load('pop_sign_tssi', data_dir=".", with_info=True)