#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import random
import pandas as pds
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from datetime import datetime

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

    def evaluate_mc_model(self, trainX, trainy, testX, testy, n_filters, n_kernel, outputtype):
        # multi-channel model, i.e. tasks are arranged in pages along the same matrix dimension
        # as channels are. thus the data matrix is of dimensions:
        # trials x time x  [ data (emg/gyro/acc) * tasks ]
        verbose, epochs, batch_size = 0, 10, 32
        n_timesteps = trainX.shape[1]
        n_features  = trainX.shape[2]

        model = Sequential()

        model.add(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu', input_shape=(n_timesteps, n_features)))
        model.add(BatchNormalization())
        #model.add(MaxPooling1D(pool_size=2))
        #model.add(Dropout(0.2))

        model.add(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu'))
        model.add(BatchNormalization())

        model.add(Conv1D(filters=math.floor(n_filters/2), kernel_size=n_kernel, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        #model.add(Dropout(0.2))        ###model.add(Flatten())

        model.add(Conv1D(filters=math.floor(n_filters/3), kernel_size=n_kernel, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        #model.add(Dropout(0.2))  ###model.add(Flatten())
        #model.add(LSTM(128, return_sequences=True))

        model.add(GlobalAveragePooling1D())
        ###model.add(Dense(100, activation='relu'))
        model.add(Dense(150, activation='relu'))
        #model.add(Dense(n_outputs, activation='softmax'))

        model.add(Dense(150, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        if outputtype == 'reg':
            model.add(Dense(1, activation="linear"))
            model.compile(loss='mean_squared_error', optimizer='adam')
            model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
            preds = model.predict(testX)
            preds = np.reshape(preds, newshape=preds.shape[0])

            datprint = pds.DataFrame()
            datprint['updrs']      = testy
            datprint['prediction'] = preds

            now     = datetime.now()
            timestr = now.strftime("%Y-%m-%d_%H:%M:%S")
            parstr  = 'filters_%d_kernel_%d' % (n_filters, n_kernel)
            filestr = timestr + '_cnn_reg_mc_' + parstr + '.csv'

            datprint.to_csv(filestr, index=False, header=True)

            self.accuracy = np.corrcoef(preds, testy)[1, 0]

        else:
           # model.add(Dense(n_outputs, activation='softmax'))
            model.add(Dense(2, activation='sigmoid'))
          #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
          # fit network
            model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
          # evaluate model
            _, self.accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=verbose)

        return self.accuracy


    def evaluate_mh_model(self, trainX, trainy, testX, testy, n_filters, n_kernel):
        ## todo setup a model in which all trials are trained separately asin: https://blog.goodaudience.com/predicting-physical-activity-based-on-smartphone-sensor-data-using-cnn-lstm-9182dd13b6bc
        verbose, epochs, batch_size = 0, 10, 32
        n_timesteps, n_features, n_outputs = trainX[0].shape[1], trainX[0].shape[2], trainy.shape[1]

        def define_head(self, trainX, trainy, n_filters, n_kernel):
            model = Sequential()

            model.add(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu',
                             input_shape=(n_timesteps, n_features), padding='same'))

            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.2))
            model.add(Conv1D(filters=math.ceil(n_filters/2), kernel_size=n_kernel, activation='relu', padding='same'))

            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.2))
            model.add(Conv1D(filters=math.ceil(n_filters/3), kernel_size=n_kernel, activation='relu', padding='same'))

            return model

        model_heads = list()
        for i in range(0, len(trainX)):
            model_heads.append(define_head(self, trainX[i], trainy, n_filters, n_kernel))

        mergedout = add([model_heads[0].output, model_heads[1].output, model_heads[2].output])
        mergedout = Flatten()(mergedout)
        mergedout = Dense(256, activation='relu')(mergedout)
        mergedout = Dropout(.5)(mergedout)
        mergedout = Dense(128, activation='relu')(mergedout)
        mergedout = Dropout(.35)(mergedout)
        mergedout = Dense(n_outputs, activation='softmax')(mergedout)

        model = Model([model_heads[0].input, model_heads[1].input, model_heads[2].input], mergedout)

        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        # fit network
        model.fit([trainX[0], trainX[1], trainX[2]], trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

        # evaluate model
        _, self.accuracy = model.evaluate([testX[0], testX[1], testX[2]],testy, batch_size=batch_size, verbose=verbose)
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
