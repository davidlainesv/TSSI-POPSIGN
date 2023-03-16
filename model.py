from config import NUM_CLASSES
import tensorflow as tf
from efficient_net_b0 import EfficientNetB0
from mobile_net_v2_model import MobileNetV2
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import DenseNet121
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.nasnet import NASNetMobile
# from tensorflow.keras.applications.efficientnet import EfficientNetB0

import wandb


def build_densenet121_model(input_shape=[None, 128, 3], dropout=0,
                            optimizer=None, pretraining=True):
    # setup model
    base_model = None
    if pretraining:
        # path_to_downloaded_file = tf.keras.utils.get_file(
        #     "tssi_densenet_wlasl100.zip",
        #     "https://storage.googleapis.com/cloud-ai-platform-f3305919-42dc-47f1-82cf-4f1a3202db74/tssi_densenet_wlasl100.zip",
        #     extract=True)
        # path_to_downloaded_file = path_to_downloaded_file.replace(".zip", "")

        # inputs = Input(shape=input_shape)
        # x = DenseNet121(input_shape=input_shape, weights=None,
        #                 include_top=False, pooling='avg')(inputs)
        # x = Dropout(0)(x)
        # base_model = Model(inputs=inputs, outputs=x)
        # base_model.load_weights(path_to_downloaded_file + "/weights")
        # # base_model.trainable = False
        # x = base_model(inputs, training=False)
        # x = Dropout(dropout)(x)
        # predictions = Dense(NUM_CLASSES, activation='softmax')(x)
        # model = Model(inputs=base_model.input, outputs=predictions)
        inputs = Input(shape=input_shape)
        x = DenseNet121(input_shape=input_shape, weights="imagenet",
                        include_top=False, pooling='avg')(inputs)
        x = Dropout(dropout)(x)
        predictions = Dense(NUM_CLASSES, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=predictions)
    else:
        inputs = Input(shape=input_shape)
        x = DenseNet121(input_shape=input_shape, weights=None,
                        include_top=False, pooling='avg')(inputs)
        x = Dropout(dropout)(x)
        predictions = Dense(NUM_CLASSES, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=predictions)

    # setup the metrics
    metrics = [
        TopKCategoricalAccuracy(k=1, name='top_1', dtype=tf.float32)
    ]

    # compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=metrics)

    return model, base_model


def build_densenet121_model(input_shape=[None, 128, 3], dropout=0,
                            optimizer=None, pretraining=True):
    # setup model
    weights = 'imagenet' if pretraining else None
    inputs = Input(shape=input_shape)
    x = DenseNet121(input_shape=input_shape, weights=weights,
                    include_top=False, pooling='avg')(inputs)
    x = Dropout(dropout)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)

    # setup the metrics
    metrics = [
        TopKCategoricalAccuracy(k=1, name='top_1', dtype=tf.float32)
    ]

    # compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=metrics)

    return model, None


def build_mobilenetv2_model(input_shape=[None, 128, 3], dropout=0,
                            optimizer=None, pretraining=True):
    # setup model
    weights = "imagenet" if pretraining else None
    inputs = Input(shape=input_shape)
    x = MobileNetV2(input_shape=input_shape, weights=weights,
                    include_top=False, pooling="avg")(inputs)
    x = Dropout(dropout)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)

    # setup the metrics
    metrics = [
        TopKCategoricalAccuracy(k=1, name='top_1', dtype=tf.float32)
    ]

    # compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=metrics)

    return model


def build_nasnetmobile_model(input_shape=[None, 128, 3], dropout=0,
                             optimizer=None, pretraining=True):
    # setup model
    weights = "imagenet" if pretraining else None
    inputs = Input(shape=input_shape)
    x = NASNetMobile(input_shape=input_shape, weights=weights,
                     include_top=False, pooling="avg")(inputs)
    x = Dropout(dropout)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)

    # setup the metrics
    metrics = [
        TopKCategoricalAccuracy(k=1, name='top_1', dtype=tf.float32)
    ]

    # compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=metrics)

    return model


def build_efficientnet_model(input_shape=[None, 128, 3], dropout=0,
                             optimizer=None, pretraining=True):
    # setup model
    weights = "imagenet" if pretraining else None
    inputs = Input(shape=input_shape)
    # inputs = tf.keras.layers.Resizing(128, 224)(inputs)
    x = EfficientNetB0(input_shape=input_shape, weights=weights,
                       include_top=False, pooling="avg")(inputs)
    x = Dropout(dropout)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)

    # setup the metrics
    metrics = [
        TopKCategoricalAccuracy(k=1, name='top_1', dtype=tf.float32)
    ]

    # compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=metrics)

    return model
