#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import random
import pandas as pds
import numpy as np
import os
import matplotlib.pyplot as plt
import math

from keras.models import Sequential, Model
from keras.layers import *
from keras.layers.convolutional import Conv1D, MaxPooling1D


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
        #model.add(Dropout(0.2))
        ###model.add(Dense(n_outputs, activation='softmax'))
        model.add(Dense(n_outputs, activation='sigmoid'))

        ###model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # fit network
        model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

        # evaluate model
        _, self.accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=verbose)
        return self.accuracy

    def evaluate_mc_model(self, trainX, trainy, testX, testy, n_filters, n_kernel):
        # multi-channel model, i.e. tasks are arranged in pages along the same matrix dimension
        # as channels are. thus the data matrix is of dimensions:
        # trials x time x  [ data (emg/gyro/acc) * tasks ]
        verbose, epochs, batch_size = 0, 10, 32
        n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
        model = Sequential()

        model.add(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu', input_shape=(n_timesteps, n_features)))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))

        model.add(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))        ###model.add(Flatten())

        model.add(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))  ###model.add(Flatten())

        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(128, return_sequences=True))

        model.add(GlobalAveragePooling1D())
        ###model.add(Dense(100, activation='relu'))
        model.add(Dense(150, activation='relu'))
        #model.add(Dense(n_outputs, activation='softmax'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        # model.add(Dense(n_outputs, activation='softmax'))
        model.add(Dense(n_outputs, activation='sigmoid'))

        ###model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # fit network
        model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

        # evaluate model
        _, self.accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=verbose)
        return self.accuracy


    def evaluate_mh_model(self, trainX, trainy, testX, testy, n_filters, n_kernel):
        ## todo setup a model in which all trials are trained separately asin: https://blog.goodaudience.com/predicting-physical-activity-based-on-smartphone-sensor-data-using-cnn-lstm-9182dd13b6bc
        verbose, epochs, batch_size = 0, 10, 32
        n_timesteps, n_features, n_outputs = trainx[0].shape[1], trainx[0].shape[2], trainy.shape[1]

        def define_head(self, trainx, trainy, n_filters, n_kernel):
            model = sequential()

            model.add(conv1d(filters=n_filters, kernel_size=n_kernel, activation='relu',
                             input_shape=(n_timesteps, n_features), padding='same'))

            model.add(batchnormalization())
            model.add(maxpooling1d(pool_size=2))
            model.add(dropout(0.2))
            model.add(conv1d(filters=math.ceil(n_filters/2), kernel_size=n_kernel, activation='relu', padding='same'))

            model.add(batchnormalization())
            model.add(maxpooling1d(pool_size=2))
            model.add(dropout(0.2))
            model.add(conv1d(filters=math.ceil(n_filters/3), kernel_size=n_kernel, activation='relu', padding='same'))

            return model

        model_heads = list()
        for i in range(0, len(trainx)):
            model_heads.append(define_head(self, trainx[i], trainy, n_filters, n_kernel))

        mergedout = add()([model_heads[0].output, model_heads[1].output, model_heads[2].output])
        mergedout = flatten()(mergedout)
        mergedout = dense(256, activation='relu')(mergedout)
        mergedout = dropout(.5)(mergedout)
        mergedout = dense(128, activation='relu')(mergedout)
        mergedout = dropout(.35)(mergedout)
        mergedout = dense(n_outputs, activation='softmax')(mergedout)

        model = model([model_heads[0].input, model_heads[1].input, model_heads[2].input], mergedout)

        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        # fit network
        model.fit([trainx[0], trainx[1], trainx[2]], trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

        # evaluate model
        _, self.accuracy = model.evaluate([testx[0], testx[1], testx[2]],testy, batch_size=batch_size, verbose=verbose)
        return self.accuracy



    def evaluate_combined_model(self, trainX, trainy, testX, testy, n_filters, n_kernel):
        verbose, epochs, batch_size = 0, 10, 32
        n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

        # reshape data into time steps of sub-sequences
        n_steps, n_length = 4, 125
        trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
        testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))

        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu'),
                                  input_shape=(None, n_length, n_features)))
        model.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu')))
        model.add(TimeDistributed(Dropout(0.5)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))

        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(n_outputs, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        # fit network
        model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

        # evaluate model
        _, self.accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=verbose)
        return self.accuracy
