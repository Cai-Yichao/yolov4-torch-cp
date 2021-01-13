# -*- coding: UTF-8 -*-
"""
2020_11_03
"""

import cv2
import numpy as np


def hsv_tuning(image, hue, sat, val):
    # 色域变换
    # hue = rand(-hue, hue)
    # sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    # val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = cv2.cvtColor(np.array(image, np.float32) / 255., cv2.COLOR_RGB2HSV)
    x[..., 0] += hue * 360
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x[:, :, 0] > 360, 0] = 360
    x[:, :, 1:][x[:, :, 1:] > 1] = 1
    x[x < 0] = 0
    image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)
    image_data = cv2.convertScaleAbs(image_data, alpha=255)
    return image_data
