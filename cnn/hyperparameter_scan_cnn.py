#!/usr/bin/env python
# -*- coding: utf-8 -*-


from keras.models import Sequential
from keras.layers import Dropout, Dense
import talos as ta
from talos.utils import lr_normalizer
from talos.metrics.keras_metrics import f1score
from keras.layers import Flatten

class ModelHyperparams:
    def __init__(self, trainX, trainy, testX, testy, params):
        pass

    def evaluate_params(self, trainX, trainy, testX, testy, params):
        # next we can build the model exactly like we would normally do it
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(32, input_dim=4, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer=params['optimizer'], loss=params['losses'])

        out = model.fit(trainX, trainy, verbose=0,
                            batch_size=params['batch_size'],
                            epochs=params['epochs'],
                            validation_data = [testX, testy])

        # finally we have to make sure that history object and model are returned
        return out, model

    def grid_search(self, trainX, trainy, testX, testy, params):
        t = ta.Scan(x=trainX,
                    y=trainy,
                    x_val=testX,
                    y_val=testy,
                    model=self.evaluate_params(),
                    grid_downsample=0.01,
                    params=params,
                    dataset_name='svmONOFF',
                    experiment_no='1')