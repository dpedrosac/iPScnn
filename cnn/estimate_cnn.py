#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import random
import pandas as pds
import numpy as np
import os
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


class ModelDefinition:
    def __init__(self, trainX, trainy, testX, testy):
        pass

    def evaluate_model(self, trainX, trainy, testX, testy, n_filters, n_kernel):
        verbose, epochs, batch_size = 0, 10, 32
        n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
        model = Sequential()

        model.add(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu', input_shape=(n_timesteps, n_features)))
        model.add(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu'))
        ###model.add(Dropout(0.5))
        ###model.add(MaxPooling1D(pool_size=2))
        model.add(MaxPooling1D(pool_size=3))
        model.add(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu'))
        model.add(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu'))
        ###model.add(Flatten())
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.2))
        ###model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(n_outputs, activation='softmax'))

        ###model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # fit network
        model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

        # evaluate model
        _, self.accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
        return self.accuracy
