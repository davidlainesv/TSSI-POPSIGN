import math
import tensorflow as tf
import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark


class PadIfLessThan(tf.keras.layers.Layer):
    def __init__(self, frames=128, **kwargs):
        super().__init__(**kwargs)
        self.frames = frames

    @tf.function
    def call(self, images):
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        height_pad = tf.math.maximum(0, self.frames - height)
        paddings = [[0, 0], [0, height_pad], [0, 0], [0, 0]]
        padded_images = tf.pad(images, paddings, "CONSTANT")
        return padded_images


class ResizeIfMoreThan(tf.keras.layers.Layer):
    def __init__(self, frames=128, **kwargs):
        super().__init__(**kwargs)
        self.frames = frames

    @tf.function
    def call(self, images):
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        new_size = [self.frames, width]
        resized = tf.cond(height > self.frames,
                          lambda: tf.image.resize(images, new_size),
                          lambda: images)
        return resized


class Center(tf.keras.layers.Layer):
    def __init__(self, around_index=0, **kwargs):
        super().__init__(**kwargs)
        self.around_index = around_index

    @tf.function
    def call(self, batch):
        # batch.shape => (examples, frames, joints, coordinates)
        # [color].shape => (examples, frames, joints)
        [red, green, blue] = tf.unstack(batch, axis=-1)

        # [color]_around_joint.shape => (examples, frames, 1)
        red_around_joint = tf.expand_dims(
            red[:, :, self.around_index], axis=-1)
        green_around_joint = tf.expand_dims(
            green[:, :, self.around_index], axis=-1)

        # new_[color].shape => (examples, frames, joints)
        new_red = red - red_around_joint
        new_green = green - green_around_joint

        return tf.stack([new_red, new_green, blue], axis=-1)


class TranslationScaleInvariant(tf.keras.layers.Layer):
    def __init__(self, level='frame', **kwargs):
        super().__init__(**kwargs)
        self.level_dict = {
            'frame': tf.constant(0),
            'joint': tf.constant(1)
        }
        self.level = self.level_dict[level]

    @tf.function
    def frame_level(self, batch):
        # batch.shape => (examples, frames, joints, coordinates)
        # [color].shape => (examples, frames, joints)
        [red, green, blue] = tf.unstack(batch, axis=-1)

        # [color]_min.shape => (examples, frames, 1)
        # min at each frame per channel
        red_min = tf.reduce_min(red, axis=-1, keepdims=True)
        green_min = tf.reduce_min(green, axis=-1, keepdims=True)
        blue_min = tf.reduce_min(blue, axis=-1, keepdims=True)

        # [color]_max.shape => (examples, frames, 1)
        # max at each frame per channel
        red_max = tf.reduce_max(red, axis=-2, keepdims=True)
        green_max = tf.reduce_max(green, axis=-2, keepdims=True)
        blue_max = tf.reduce_max(blue, axis=-2, keepdims=True)

        # [color]_dist.shape => (examples, frames, 1)
        # distance between max and min at each frame per channel
        red_dist = red_max - red_min
        green_dist = green_max - green_min
        blue_dist = blue_max - blue_min

        # [color]_dist_max.shape => ()
        # max_distance of all frames per channel
        red_dist_max = tf.reduce_max(red_dist)
        green_dist_max = tf.reduce_max(green_dist)
        blue_dist_max = tf.reduce_max(blue_dist)

        # new_[color].shape => (examples, frames, joints)
        new_red = tf.math.divide_no_nan((red - red_min), red_dist_max)
        new_green = tf.math.divide_no_nan((green - green_min), green_dist_max)
        new_blue = tf.math.divide_no_nan((blue - blue_min), blue_dist_max)

        return tf.stack([new_red, new_green, new_blue], axis=-1)

    @tf.function
    def joint_level(self, batch):
        # batch.shape => (examples, frames, joints, coordinates)
        # [color].shape => (examples, frames, joints)
        [red, green, blue] = tf.unstack(batch, axis=-1)

        # [color]_min.shape => (examples, 1, joints)
        # min at each joint per channel
        red_min = tf.reduce_min(red, axis=-2, keepdims=True)
        green_min = tf.reduce_min(green, axis=-2, keepdims=True)
        blue_min = tf.reduce_min(blue, axis=-2, keepdims=True)

        # [color]_max.shape => (examples, 1, joints)
        # max at each joint per channel
        red_max = tf.reduce_max(red, axis=-2, keepdims=True)
        green_max = tf.reduce_max(green, axis=-2, keepdims=True)
        blue_max = tf.reduce_max(blue, axis=-2, keepdims=True)

        # [color]_dist.shape => (examples, 1, joint)
        # distance between max and min at each joint per channel
        red_dist = red_max - red_min
        green_dist = green_max - green_min
        blue_dist = blue_max - blue_min

        # [color]_dist_max.shape => ()
        # max_distance of all joints per channel
        red_dist_max = tf.reduce_max(red_dist)
        green_dist_max = tf.reduce_max(green_dist)
        blue_dist_max = tf.reduce_max(blue_dist)

        # new_[color].shape => (examples, frames, joints)
        new_red = tf.math.divide_no_nan((red - red_min), red_dist_max)
        new_green = tf.math.divide_no_nan((green - green_min), green_dist_max)
        new_blue = tf.math.divide_no_nan((blue - blue_min), blue_dist_max)

        return tf.stack([new_red, new_green, new_blue], axis=-1)

    @tf.function
    def call(self, batch):
        batch = tf.cond(
            self.level == self.level_dict['frame'],
            lambda: self.frame_level(batch),
            lambda: self.joint_level(batch))
        return batch


class FillBlueWithAngle(tf.keras.layers.Layer):
    def __init__(self, x_channel=0, y_channel=1, scale_to=[0, 1], **kwargs):
        super().__init__(**kwargs)
        self.x_channel = x_channel
        self.y_channel = y_channel
        self.scale_to = scale_to

    @tf.function
    def call(self, batch):
        batch = tf.cast(batch, tf.float32)
        unstacked = tf.unstack(batch, axis=-1)
        x, y = unstacked[self.x_channel], unstacked[self.y_channel]
        angles = tf.math.atan2(y, x) * (180 / math.pi) % 360
        data_min, data_max = 0, 359
        range_min, range_max = self.scale_to
        std = (angles - data_min) / (data_max - data_min)
        scaled = std * (range_max - range_min) + range_min

        return tf.stack([x, y, scaled], axis=-1)


class FillZWithZeros(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf.function
    def call(self, batch):
        [red, green, _] = tf.unstack(batch, axis=-1)
        zeros = tf.zeros(tf.shape(red), dtype=red.dtype)
        return tf.stack([red, green, zeros], axis=-1)
