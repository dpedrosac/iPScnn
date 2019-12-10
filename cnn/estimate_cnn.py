#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import random
import pandas as pds
import numpy as np
import os
import matplotlib.pyplot as plt
from random import seed

from keras.models import Sequential, Input, Model
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling1D, TimeDistributed, LSTM, BatchNormalization, Concatenate
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

    def evaluate_alt_model(self, trainX, trainy, testX, testy, n_filters, n_kernel):
        ## TODO setup a model in which all trials are trained separately asin: https://blog.goodaudience.com/predicting-physical-activity-based-on-smartphone-sensor-data-using-cnn-lstm-9182dd13b6bc
        verbose, epochs, batch_size = 0, 10, 32
        n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
        model = Sequential()

        model.add(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu',
                         input_shape=(n_timesteps, n_features), padding='same'))

        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))
        model.add(Conv1D(filters=n_filters/2, kernel_size=n_kernel, activation='relu', padding='same'))

        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))
        model.add(Conv1D(filters=n_filters/2, kernel_size=n_kernel, activation='relu', padding='same'))

        model.add(BatchNormalization())
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(128))
        model.add(Dropout(0.2))

        model.add(Dense(n_outputs, activation='sigmoid'))
        model.add(BatchNormalization())
        ###model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        # fit network
        model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

        # evaluate model
        _, self.accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=verbose)
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


    def evaluate_split_model(self, trainX, trainy, testX, testy, n_filters, n_kernel):
        verbose, epochs, batch_size = 0, 10, 32

        n_features, n_steps = 1, 500
        # separate input data
        trainX1 = trainX[:, :, 0].reshape(trainX.shape[0], trainX.shape[1], n_features)
        trainX2 = trainX[:, :, 1].reshape(trainX.shape[0], trainX.shape[1], n_features)
        trainX3 = trainX[:, :, 2].reshape(trainX.shape[0], trainX.shape[1], n_features)

        n_timesteps, n_features, n_outputs = trainX1.shape[1], trainX1.shape[2], trainy.shape[1]

        model1 = Sequential()
        model1.add(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu',
                         input_shape=(n_timesteps, n_features), padding='same'))
        model1.add(MaxPooling1D(pool_size=3))
        model1.add(Dropout(0.2))
        model1.add(BatchNormalization())
        model1.add(MaxPooling1D(pool_size=2))
        model1.add(Dropout(0.2))


        model2 = Sequential()
        model2.add(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu',
                         input_shape=(n_timesteps, n_features), padding='same'))
        model2.add(MaxPooling1D(pool_size=3))
        model2.add(Dropout(0.2))
        model2.add(BatchNormalization())
        model2.add(MaxPooling1D(pool_size=2))
        model2.add(Dropout(0.2))

        merged = Concatenate([model1, model2], name='Concatenate')

        final_model_output = Dense(3, activation='sigmoid')(merged)
        final_model = Model(inputs=[model1, model2], outputs=final_model_output,
                            name='Final_output')
        final_model.compile(optimizer='adam', loss='binary_crossentropy')
        # To train
        final_model.fit([Left_data, Right_data], labels, epochs=10, batch_size=32)


        model_cnn = Sequential()
        model_cnn.add(Concatenate([model1, model2]))
        model_cnn.add(Dense(1, init='normal', activation='sigmoid'))

        model_cnn.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        seed(2019)

        # fit network
        model_cnn.fit([trainX1, trainX2], trainy, batch_size=batch_size, nb_epoch=epochs, verbose=1)

        # evaluate model
        _, self.accuracy = model_cnn.evaluate(testX, testy, batch_size=batch_size, verbose=verbose)
        return self.accuracy



