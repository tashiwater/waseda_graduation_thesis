#!/usr/bin/env python
# coding: utf-8

import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
import torch
import torchvision


class OnlineDataSet:
    def __init__(self, cae, device, high_freq):
        rospy.Subscriber("/image_raw", Image, self.cb_image)
        rospy.Subscriber(
            "/touchence/sensor_data", Float32MultiArray, self.TouchSensorCallback
        )
        rospy.Subscriber(
            "/torobo/joint_state_server/joint_states", JointState, self.ToroboCallback
        )
        self._bridge = CvBridge()
        self._imgs = []
        self._tactiles = []
        self._positions = []
        self._efforts = []
        self._high_freq = high_freq

        self._cae = cae
        self._cae.to(device)
        self._cae.eval()
        self._device = device

        self._position_before_scale = [
            [-1.309, 4.451],
            [-2.094, 0.140],
            [-2.880, 2.880],
            [-0.524, 2.269],
            [-2.880, 2.880],
            [-1.396, 1.396],
            [-1.396, 0.087],
        ]
        self._effort_before_scale = [
            [-2, 40],
            [-40, 15],
            [-5, 15],
            [-10, 15],
            [-5, 5],
            [-5, 5],
            [-5, 5],
        ]

    def cal_high_freq(self):
        self._tactiles.append(self._last_tactile)
        self._positions.append(self._last_motor.position)
        self._efforts.append(self._last_motor.position)

    def cal_features(self):

        # log
        self.cal_high_freq()
        self._imgs.append(img)
        # extract feature
        position = self._positions[-self._high_freq :].mean(axis=0)
        effort = self._tactiles[-self._high_freq :].mean(axis=0)
        tactile = self._tactiles[-self._high_freq :].mean(axis=0)
        img_feature = self.get_img_feature(img)

        tactile_before_scale = [[0, 1] for _ in range(tactile)]
        image_before_scale = [[0, 1] for _ in range(image_feature)]

        position = self.normalize(position, self._position_before_scale)
        effort = self.normalize(effort, self._effort_before_scale)
        tactile = self.normalize(tactile, tactile_before_scale)
        img_feature = self.normalize(img_feature, img_before_scale)

        connected_data = np.block(
            [motion_preprocessed, tactile_preprocessed, image_feature_data]
        )
        self._connected_data = torch.tensor(connected_data)

    def get_connected_data(self):
        return self._connected_data

    def cb_image(self, data):
        self._last_img = self._bridge.imgmsg_to_cv2(data, "rgb8")

    def TouchSensorCallback(self, data):
        self._last_tactile = data.data

    def ToroboCallback(self, data):
        self._last_motor = data.data

    def get_img_feature(self, img):
        img_tensor = torchvision.transforms.ToTensor()(img)
        img_tensor = img_tensor.to(self._device)
        img_feature = self._cae.encoder(img_tensor)
        return img_feature.to("cpu").detach().numpy()

    def _each_normalization(self, data, indataRange, outdataRange=[-0.9, 0.9]):
        data = (data - indataRange[0]) / (indataRange[1] - indataRange[0])
        data = data * (outdataRange[1] - outdataRange[0]) + outdataRange[0]
        return data
,    def normalize(self, data, before_scale, after_scale = [-0.9,0.9]):
        for i, scale in enumerate(before_scale):
            data[i] = self._each_normalization(data[i], scale, after_scale)
        return data

    def get_header(self, add_word):
        return (
            [add_word + "position{}".format(i) for i in range(7)]
            + [add_word + "torque{}".format(i) for i in range(7)]
            + [add_word + "tactile{}".format(i) for i in range(16)]
            + [add_word + "image{}".format(i) for i in range(20)]
        )
    
    def reverse_position(self, position):
        return self.normalize(position, [-0.9, 0.9], self._position_before_scale)

