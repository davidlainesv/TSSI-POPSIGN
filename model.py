from config import NUM_CLASSES
import tensorflow as tf
from efficient_net_b0 import EfficientNetB0
from densenet import DenseNet121
from loss import select_loss
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras import Input
from tensorflow.keras.models import Model
# from tensorflow.keras.applications.densenet import DenseNet121
# from tensorflow.keras.applications.efficientnet import EfficientNetB0
import wandb


def build_densenet121_model(input_shape=[None, 135, 2],
                            dropout=0,
                            optimizer=None,
                            pretraining=False,
                            use_loss="crossentroypy",
                            growth_rate=12,
                            attention=None):
    if pretraining and growth_rate != 32 and attention != None:
        raise Exception(
            "pretraining on ImageNet is also compatible to growth_rate=32 and attention=None")

    # setup backbone
    weights_dir = None
    if pretraining:
        api = wandb.Api()
        # get the directory in which the model is saved
        weights_at = api.artifact(
            'davidlainesv/autsl-testing/run_fz31vdq1_model:v0')
        # download the directory in which the model is saved
        weights_dir = weights_at.download()

    backbone = DenseNet121(input_shape=input_shape,
                           weights=None,
                           include_top=False,
                           pooling='avg',
                           growth_rate=growth_rate,
                           attention=attention)

    # setup model
    inputs = Input(shape=input_shape)
    x = backbone(inputs)
    x = Dropout(dropout)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)

    # setup the metrics
    metrics = [
        TopKCategoricalAccuracy(k=1, name='top_1', dtype=tf.float32)
    ]

    # setup the loss
    loss = select_loss(use_loss)

    # compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # load weights
    if pretraining:
        model.load_weights(weights_dir + "/weights")

    return model


def build_efficientnet_model(input_shape=[None, 128, 3],
                             dropout=0,
                             optimizer=None,
                             pretraining=True,
                             use_loss="crossentropy"):
    # setup backbone
    weights = "imagenet" if pretraining else None
    backbone = EfficientNetB0(input_shape=input_shape,
                              weights=weights,
                              include_top=False,
                              pooling="avg")

    # setup model
    inputs = Input(shape=input_shape)
    x = backbone(inputs)
    x = Dropout(dropout)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)

    # setup the metrics
    metrics = [
        TopKCategoricalAccuracy(k=1, name='top_1', dtype=tf.float32)
    ]

    # setup the model
    loss = select_loss(use_loss)

    # compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
