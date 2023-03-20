import tensorflow as tf
from mediapipe.python.solutions.pose import PoseLandmark


class Preprocessing(tf.keras.layers.Layer):
    # original order = ['face', 'left_hand', 'pose', 'right_hand']

    range_dict = {
        'face': range(0, 468),
        'leftHand': range(468, 468+21),
        'pose': range(468+21, 468+21+33),
        'rightHand': range(468+21+33, 468+21+33+21),
        'root': range(468+21+33+21, 468+21+33+21+1)
    }

    slice_dict = {
        'face': slice(0, 468),
        'leftHand': slice(468, 468+21),
        'pose': slice(468+21, 468+21+33),
        'rightHand': slice(468+21+33, 468+21+33+21),
        'root': slice(468+21+33+21, 468+21+33+21+1)
    }

    def __init__(self, tssi_order, **kwargs):
        super().__init__(**kwargs)
        joints_idxs = []
        for joint in tssi_order:
            joint_type = joint.split("_")[0]
            if joint_type == "root":
                landmark_id = 0
            else:
                landmark_id = int(joint.split("_")[1])
            idx = self.range_dict[joint_type][landmark_id]
            joints_idxs.append(idx)
        self.joints_idxs = joints_idxs
        self.nose_idx = self.range_dict["pose"][PoseLandmark.NOSE]
        self.left_wrist_idx = self.range_dict["pose"][PoseLandmark.LEFT_WRIST]
        self.right_wrist_idx = self.range_dict["pose"][PoseLandmark.RIGHT_WRIST]
        self.root_idx = self.range_dict["root"][0]

    @tf.function
    def call(self, keypoints):
        # keypoints = self.batch_if_necessary(keypoints)
        keypoints = self.fill_z_with_zeros(keypoints)
        keypoints = self.fill_nan_values(keypoints)
        keypoints = self.add_root(keypoints)
        keypoints = self.sort_columns(keypoints)
        return keypoints

    @tf.function
    def batch_if_necessary(self, keypoints):
        # usually keypoints.shape = (frames, joints, channels)
        ndims = tf.size(tf.shape(keypoints))
        keypoints = tf.cond(tf.equal(ndims, 3),
                            tf.expand_dims(keypoints, 0),
                            keypoints)
        return keypoints

    @tf.function
    def fill_nan_values(self, keypoints):
        face = keypoints[:, :, self.slice_dict["face"], :]
        left_hand = keypoints[:, :, self.slice_dict["leftHand"], :]
        body = keypoints[:, :, self.slice_dict["pose"], :]
        right_hand = keypoints[:, :, self.slice_dict["rightHand"], :]

        nose = keypoints[:, :, self.nose_idx, :]
        nose = tf.expand_dims(nose, axis=2)
        left_wrist = keypoints[:, :, self.left_wrist_idx, :]
        left_wrist = tf.expand_dims(left_wrist, axis=2)
        right_wrist = keypoints[:, :, self.right_wrist_idx, :]
        right_wrist = tf.expand_dims(right_wrist, axis=2)

        left_hand = tf.where(
            tf.math.is_nan(left_hand),
            tf.repeat(left_wrist, 21, axis=2),
            left_hand)
        right_hand = tf.where(
            tf.math.is_nan(right_hand),
            tf.repeat(right_wrist, 21, axis=2),
            right_hand)
        face = tf.where(
            tf.math.is_nan(face),
            tf.repeat(nose, 468, axis=2),
            face)

        keypoints = tf.concat([face, left_hand, body, right_hand], axis=2)

        return keypoints

    @tf.function
    def fill_z_with_zeros(self, keypoints):
        x, y, _ = tf.unstack(keypoints, axis=-1)
        zeros = tf.zeros(tf.shape(x), x.dtype)
        return tf.stack([x, y, zeros], axis=-1)

    @tf.function
    def add_root(self, keypoints):
        left = keypoints[:, :, self.left_wrist_idx, :]
        right = keypoints[:, :, self.right_wrist_idx, :]
        root = (left + right) / 2
        root = tf.expand_dims(root, axis=2)
        keypoints = tf.concat([keypoints, root], axis=2)
        return keypoints

    @tf.function
    def sort_columns(self, keypoints):
        keypoints = tf.gather(keypoints, indices=self.joints_idxs, axis=2)
        return keypoints
