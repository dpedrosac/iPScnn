#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob, yaml, getpass
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Sized
from scipy.special import comb
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.signal import medfilt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
import matplotlib.pyplot as plt
import ast
import math
from numpy.lib.stride_tricks import as_strided

import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UserWarning, message='Variables are collinear')

__all__ = ["file_browser", "moving_window_stride", "window_trapezoidal", "as_strided"]


def __init__(self):
    with open('/media/storage/iPScnn/config.yaml', 'r') as f:
        d = yaml.load(f.read())

    # TODO insert a list of features to extract in the yaml file
    self.debug = False
    self.samplerate = 200
    self.emgdir = d[0]['dataworks']['folders'][getpass.getuser()]['emgrawdir']


def file_browser(self, word):
    """function which helps to find files with certain content, specified in the input"""
    file = []
    os.chdir(self.emgdir)

    for f in glob.glob("*"):
        if word in f:
            file.append(f)

    return file

def convert_types_in_dict(xml_dict):
    """
    Evaluates all dictionary entries into Python literal structure, as dictionary read from XML file is always string.
    If value can not be converted it passed as it is.
    :param xml_dict: Dict - Dictionary of XML entries
    :return: Dict - Dictionary with converted values
    """
    out = {}
    for el in xml_dict:
        try:
            out[el] = ast.literal_eval(xml_dict[el])
        except ValueError:
            out[el] = xml_dict[el]

    return out


def moving_window_stride(array, window, step):
    """
    Returns view of strided array for moving window calculation with given window size and step
    :param array: numpy.ndarray - input array
    :param window: int - window size
    :param step: int - step lenght
    :return: strided: numpy.ndarray - view of strided array, index: numpy.ndarray - array of indexes
    """
    stride = array.strides[0]
    win_count = math.floor((len(array) - window + step) / step)
    strided = as_strided(array, shape=(win_count, window), strides=(stride * step, stride))
    index = np.arange(window - 1, window + (win_count - 1) * step, step)
    return strided, index


def window_trapezoidal(size, slope):
    """
    Return trapezoidal window of length size, with each slope occupying slope*100% of window
    :param size: int - window length
    :param slope: float - trapezoid parameter, each slope occupies slope*100% of window
    :return: numpy.ndarray - trapezoidal window
    """
    if slope > 0.5:
        slope = 0.5
    if slope == 0:
        return np.full(size, 1)
    else:
        return np.array([1 if ((slope * size <= i) & (i <= (1 - slope) * size)) else (1 / slope * i / size) if (
                    i < slope * size) else (1 / slope * (size - i) / size) for i in range(1, size + 1)])
