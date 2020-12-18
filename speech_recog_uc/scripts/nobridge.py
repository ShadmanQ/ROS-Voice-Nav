#! /usr/bin/env python3

# Authors: Daniel Mallia, Jaime Canizales and Shadman Quazi
# Date Created: 10/14/2019
#
# ROS was written for use with Python 2 and the same holds true for the
# cv_bridge module, which compiles for use with Python 2 by default.
# (Instructions to compile for Python 3 are offered at the link below - they
# remain untested). Furthermore, it appears that cv_bridge ships hardcoded to
# support OpenCV 3.2 because this is the version supported by ROS Melodic, and
# this version seems to predate the dnn (Deep Neural Network) module. Given that
# it is easier to simply rewrite needed functions as required, rather than deal
# with tedious system configuration, the functions contained in this file
# consitute such an effort. Together they offer a way to work with ROS image
# messages and current versions of OpenCV without cv_bridge - that is...working
# with "nobridge"!

# Links referenced above:
# https://stackoverflow.com/questions/49221565/unable-to-use-cv-bridge-with-ros-kinetic-and-python3
# https://answers.ros.org/question/323897/why-cv_bridge-uses-opencv32-in-ros-melodic/

# To do list
# Implement conversions from depth to msgs for publishing?
# Introduce function overloading to allow for easier calls on depth messages.
# Add normalization for uint16 depth images to range 0 to 255?

import copy
import numpy as np
import cv2 as cv
from sensor_msgs.msg import Image

# Conversion from a ROS RGB image message (unsigned 8 bit integers) to an OpenCV
# (numpy) format
def msg_to_RGB(msg):
    image = np.frombuffer(msg.data, np.uint8)
    image = image.reshape(msg.height, msg.width, 3) # New objects made because
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)    # many of the numpy methods
    return image                                    # do not modify in-place.

# Conversion from an OpenCV (numpy) format to a ROS RGB image message (unsigned
# 8 bit integers)
def RGB_to_msg(image, source_Msg):

    conversion = cv.cvtCOLOR(image, cv.COLOR_RGB2BGR)
    conversion = conversion.flatten()
    conversion = conversion.tobytes()

    msg = copy.deepcopy(source_Msg)
    msg.data = conversion

    return msg

# Conversion from a ROS depth image message (32-bit float) to an OpenCV (numpy)
# format
def msg_to_depth_f32(msg):
    depth = np.frombuffer(msg.data, np.float32)
    depth = depth.reshape(msg.height, msg.width)
    return depth

# Conversion from a ROS depth image message (unsigned 16 bit integers) to an
# OpenCV (numpy) format
def msg_to_depth_uint16(msg):
    depth = np.frombuffer(msg.data, np.uint16)
    depth = depth.reshape(msg.height, msg.width)
    return depth

# Conversion from an OpenCV (numpy) format to a ROS depth image message (32-bit float)
def depth_f32_to_msg(depth_image, source_Msg):

    conversion = depth_image.flatten()
    conversion = conversion.tobytes()

    msg = copy.deepcopy(source_Msg)
    msg.data = conversion

    return msg