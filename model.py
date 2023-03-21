from config import NUM_CLASSES
import tensorflow as tf
from efficient_net_b0 import EfficientNetB0
from densenet import DenseNet121
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras import Input
from tensorflow.keras.models import Model
# from tensorflow.keras.applications.densenet import DenseNet121
# from tensorflow.keras.applications.efficientnet import EfficientNetB0


def focal_loss(gamma=2., alpha=4.):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002
        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]
        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})
        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        # y_true = tf.convert_to_tensor(y_true, tf.float32)
        # y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.math.add(y_pred, epsilon)
        ce = tf.math.multiply(y_true, -tf.math.log(model_out))
        weight = tf.math.multiply(y_true, tf.math.pow(
            tf.math.subtract(1., model_out), gamma))
        fl = tf.math.multiply(alpha, tf.math.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed


def build_densenet121_model(input_shape=[None, 135, 2], dropout=0,
                            optimizer=None, pretraining=True, use_focal_loss=False):
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

    # setup the loss
    if use_focal_loss:
        loss = focal_loss(alpha=1)
    else:
        loss = "categorical_crossentropy"

    # compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def build_efficientnet_model(input_shape=[None, 128, 3], dropout=0,
                             optimizer=None, pretraining=True, use_focal_loss=False):
    # setup model
    weights = "imagenet" if pretraining else None
    inputs = Input(shape=input_shape)
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
    if use_focal_loss:
        loss = focal_loss(alpha=1)
    else:
        loss = "categorical_crossentropy"
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
