# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""DenseNet models for Keras.

Reference:
  - [Densely Connected Convolutional Networks](
      https://arxiv.org/abs/1608.06993) (CVPR 2017)
"""

import tensorflow.compat.v2 as tf

from keras import backend
from keras.applications import imagenet_utils
from keras.engine import training
from keras.layers import VersionAwareLayers
from keras.utils import data_utils
from keras.utils import layer_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export

from attention_module import attach_attention_module

layers = VersionAwareLayers()


def bottleneck_layers(x, growth_rate, name, dropout=0):
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
    x1 = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + "_0_bn"
    )(x)
    x1 = layers.Activation("relu", name=name + "_0_relu")(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1, use_bias=False,
                       name=name + "_1_conv")(x1)
    # x1 = layers.Dropout(dropout)(x1) if dropout else x1
    x1 = layers.SpatialDropout2D(dropout)(x1) if dropout else x1
    return x1


def conv_block(x, growth_rate, name, dropout=0):
    """A building block for a dense block.

    Args:
      x: input tensor.
      growth_rate: float, growth rate at dense layers.
      name: string, block label.

    Returns:
      Output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
    x1 = bottleneck_layers(x, growth_rate, name, dropout=dropout)
    x1 = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn"
    )(x1)
    x1 = layers.Activation("relu", name=name + "_1_relu")(x1)
    x1 = layers.Conv2D(
        growth_rate, 3, padding="same", use_bias=False, name=name + "_2_conv"
    )(x1)
    # x1 = layers.Dropout(dropout)(x1) if dropout else x1
    x1 = layers.SpatialDropout2D(dropout)(x1) if dropout else x1
    x = layers.Concatenate(axis=bn_axis, name=name + "_concat")([x, x1])
    return x


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, *, growth_rate, name, dropout=0):
        """A building block for a dense block.

        Args:
          x: input tensor.
          growth_rate: float, growth rate at dense layers.
          name: string, block label.

        Returns:
          Output tensor for the block.
        """
        self.bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
        self.bottleneck_layers = bottleneck_layers(growth_rate, name, dropout=dropout)
        self.bn = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn")
        self.activation = layers.Activation("relu", name=name + "_1_relu")
        self.conv2d = ayers.Conv2D(growth_rate, 3, padding="same", use_bias=False, name=name + "_2_conv")
        self.dropout = layers.SpatialDropout2D(dropout)(x1) if dropout else None
        self.concat = layers.Concatenate(axis=bn_axis, name=name + "_concat")
    
    def call(self, x):
        x1 = self.bottleneck_layers(x)
        x1 = self.bn(x1)
        x1 = self.activation(x1)
        x1 = self.conv2d(x1)
        # x1 = layers.Dropout(dropout)(x1) if dropout else x1
        x1 = self.dropout(x1) if self.dropout else x1
        x = self.concat([x, x1])
        return x


def dense_block(x, blocks, growth_rate, name, attention=None, dropout=0):
    """A dense block.

    Args:
      x: input tensor.
      blocks: integer, the number of building blocks.
      name: string, block label.

    Returns:
      Output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, growth_rate, dropout=dropout,
                       name=name + "_block" + str(i + 1))

    # attention
    x = attach_attention_module(x, attention) if attention else x

    return x


def transition_block(x, reduction, name, attention=None, dropout=0):
    """A transition block.

    Args:
      x: input tensor.
      reduction: float, compression rate at transition layers.
      name: string, block label.

    Returns:
      output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + "_bn"
    )(x)
    x = layers.Activation("relu", name=name + "_relu")(x)
    x = layers.Conv2D(
        int(backend.int_shape(x)[bn_axis] * reduction),
        1,
        use_bias=False,
        name=name + "_conv",
    )(x)
    x = layers.Dropout(dropout)(x) if dropout else x
    # x = layers.AveragePooling2D(2, strides=2, name=name + "_pool")(x)

    # attention
    x = attach_attention_module(x, attention) if attention else x

    return x


class DenseNet121(tf.keras.layers.Layer):
    def __init__(self, *,
        blocks,
        pooling=None,
        growth_rate=12,
        attention=None,
        dropout=0
    ):
        """Instantiates the DenseNet architecture.

        Reference:
        - [Densely Connected Convolutional Networks](
            https://arxiv.org/abs/1608.06993) (CVPR 2017)

        This function returns a Keras image classification model,
        optionally loaded with weights pre-trained on ImageNet.

        For image classification use cases, see
        [this page for detailed examples](
          https://keras.io/api/applications/#usage-examples-for-image-classification-models).

        For transfer learning use cases, make sure to read the
        [guide to transfer learning & fine-tuning](
          https://keras.io/guides/transfer_learning/).

        Note: each Keras Application expects a specific kind of input preprocessing.
        For DenseNet, call `tf.keras.applications.densenet.preprocess_input` on your
        inputs before passing them to the model.
        `densenet.preprocess_input` will scale pixels between 0 and 1 and then
        will normalize each channel with respect to the ImageNet dataset statistics.

        Args:
          blocks: numbers of building blocks for the four dense layers.
          include_top: whether to include the fully-connected
            layer at the top of the network.
          pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.

        Returns:
          A `keras.Model` instance.
        """
        super(DenseNet121, self).__init__()
        self.blocks = blocks
        self.pooling = pooling
        self.growth_rate = growth_rate
        self.attention = attention
        self.dropout = dropout
        
        bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
        
        self.padding1 = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))
        self.conv1_conv = layers.Conv2D(32, (1, 7), strides=1, use_bias=False, name="conv1/conv") # original strides=2, padding="valid"
        self.conv1_bn = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="conv1/bn")
        self.conv1_relu = layers.Activation("relu", name="conv1/relu")
        self.padding2 = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))
        self.pool1 = layers.AveragePooling2D(3, strides=2, name="pool1")
        # self.pool1 = layers.MaxPooling2D(3, strides=2, name="pool1")
        self.last_conv = layers.Conv2D(1, 1, padding="same", use_bias=False, name="last_conv")
        self.last_bn = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="last_bn")
        self.last_relu = layers.Activation("relu", name="last_relu")
        
        self.last_pool = layers.AveragePooling2D((1, 35), strides=1, name="last_pool")
        
        self.conv2_conv = layers.Conv2D(64, (1, 7), strides=1, use_bias=False, name="conv2/conv") # original strides=2, padding="valid"
        self.conv2_bn = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="conv2/bn")
        self.conv2_relu = layers.Activation("relu", name="conv2/relu")
        
        self.conv3_conv = layers.Conv2D(128, (1, 7), strides=1, use_bias=False, name="conv3/conv") # original strides=2, padding="valid"
        self.conv3_bn = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="conv3/bn")
        self.conv3_relu = layers.Activation("relu", name="conv3/relu")
        
        self.spatial_drop = layers.SpatialDropout2D(dropout)
        
    
    def call(self, inputs):
        # x = self.padding1(inputs)
        x = self.conv1_conv(inputs)
        x = self.conv1_bn(x)
        x = self.conv1_relu(x)
        
        x = self.spatial_drop(x)
        
        x = self.conv2_conv(x)
        x = self.conv2_bn(x)
        x = self.conv2_relu(x)
        
        x = self.spatial_drop(x)
        
        x = self.conv3_conv(x)
        x = self.conv3_bn(x)
        x = self.conv3_relu(x)
        
        x = self.spatial_drop(x)
        
        # x = self.padding2(x)
        # x = self.pool1(x)

        # 3 inner blocks
        # for i in range(1):
        #     x = dense_block(x,
        #                     blocks=self.blocks[i],
        #                     growth_rate=self.growth_rate,
        #                     name=f"conv{i+2}",
        #                     attention=None,
        #                     dropout=0)
        #     x = transition_block(x,
        #                          reduction=0.5,
        #                          name=f"pool{i+2}",
        #                          attention=self.attention,
        #                          dropout=0)

        # last block
        # x = dense_block(x, blocks[3],
        #                 growth_rate,
        #                 name=f"conv5",
        #                 attention=None,
        #                 dropout=dropout)
        
        x = self.last_conv(x)
        # x = self.last_pool(x)
        x = self.last_bn(x)
        x = self.last_relu(x)

        return x

# def DenseNet121(
#     include_top=True,
#     weights="imagenet",
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000,
#     classifier_activation="softmax",
#     growth_rate=12,
#     attention=None,
#     dropout=0
# ):
#     """Instantiates the Densenet121 architecture."""
#     return DenseNet(
#         [6, 12, 24, 16],
#         include_top,
#         weights,
#         input_tensor,
#         input_shape,
#         pooling,
#         classes,
#         classifier_activation,
#         growth_rate,
#         attention,
#         dropout
#     )


