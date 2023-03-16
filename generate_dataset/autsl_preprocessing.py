import tensorflow as tf
import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark

class Preprocessing:
    range_dict = {
        'pose': range(0, 17),
        'face': range(33, 33+468),
        'leftHand': range(33+468, 33+468+21),
        'rightHand': range(33+468+21, 33+468+21+21),
        'root': range(33+468+21+21, 33+468+21+21+1)
    }
    
    def __init__(self, tssi_order):
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
        self.left_shoulder_idx = self.range_dict["pose"][PoseLandmark.LEFT_SHOULDER]
        self.right_shoulder_idx = self.range_dict["pose"][PoseLandmark.RIGHT_SHOULDER]
        self.root_idx = self.range_dict["root"][0]
        
    def __call__(self, pose):
        pose = self.reshape(pose)
        pose = self.fill_z_with_depth(pose)
        pose = self.normalize(pose)
        pose = self.add_root(pose)
        pose = self.sort_columns(pose)
        return pose
    
    def reshape(self, pose):
        pose = pose[:, 0, :, :]
        return pose
    
    def fill_z_with_depth(self, pose):
        x, y, _, depth = tf.unstack(pose, axis=-1)
        return tf.stack([x, y, depth], axis=-1)
    
    def normalize(self, pose):
        x, y, z = tf.unstack(pose, axis=-1)
        
#         x_left = x[:, self.left_shoulder_idx]
#         x_right = x[:, self.right_shoulder_idx]
#         x_mid_chest = (x_left + x_right) / 2
#         x_mid_chest = x_mid_chest[:, tf.newaxis]
#         x = x / x_mid_chest
        
#         y_left = y[:, self.left_shoulder_idx]
#         y_right = y[:, self.right_shoulder_idx]
#         y_mid_chest = (y_left + y_right) / 2
#         y_mid_chest = y_mid_chest[:, tf.newaxis]
#         y = y / y_mid_chest
        
        x = x / 512
        y = y / 512
        
        z = z / 255
        
        return tf.stack([x, y, z], axis=-1)
        
    def add_root(self, pose):
        left = pose[:, self.left_shoulder_idx, :]
        right = pose[:, self.right_shoulder_idx, :]
        root = (left + right) / 2
        root = root[:, tf.newaxis, :]
        pose = tf.concat([pose, root], axis=1)
        return pose
    
    def sort_columns(self, pose):
        pose = tf.gather(pose, indices=self.joints_idxs, axis=1)
        return pose