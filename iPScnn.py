#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tsfresh
import os
import preprocess.dataworks
import cnn.estimate_cnn
import cnn.hyperparameter_scan_cnn
from importlib import reload
from keras.losses import binary_crossentropy, categorical_crossentropy, logcosh
from keras.activations import relu, elu, sigmoid
from keras.optimizers import Adam, Nadam, RMSprop
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Dense
import talos as ta
from talos.utils import lr_normalizer
from talos.metrics.keras_metrics import f1score
from keras.layers import Flatten, Conv1D
import getpass
import pandas as pds

reload(cnn.estimate_cnn)
reload(cnn.hyperparameter_scan_cnn)

# loads and preprocesses data as needed
#datobj = preprocess.dataworks.DataPipeline('', 'all', 'all', 'all',ignbad=True) <- older version, may be deleted when settings from config.yaml works
datobj = preprocess.dataworks.DataPipeline('', '', '')
datobj.generate_subjlist()

if getpass.getuser() == 'urs':
    wdir = '/home/urs/sync/projects/autostim'
else:
    wdir = '/media/storage/iPScnn'

# the next fdew lines load the data if already extracted ot extract the data if not available.
if not os.path.isfile(os.path.join(wdir+"/data/dataON_ACCx.csv")):
    datON, datOFF, detailsON, detailsOFF = datobj.load_all(window=False)

    list_ch = ['ACCx', 'ACCy', 'ACCz', 'GYRx', 'GYRy', 'GYRz']
    ## TODO so far it only works when all data is entered. Tow possible solutions are feasible 1) saving channel data
    # into the matrix in dataworks, or 2) extracting  all data and later subselecting the channels of interest (my favorite!!)
    for idx, n in enumerate(list_ch):
        np.savetxt(os.path.join(wdir +"/data/dataON_" + list_ch[idx] + ".csv"), datON[:,:,idx], delimiter=";")
        np.savetxt(os.path.join(wdir + "/data/dataOFF_" + list_ch[idx] + ".csv"), datOFF[:, :, idx], delimiter=";")

    detailsON.to_csv(os.path.join(wdir +"/data/detailsON.csv"), mode='w', header=True)
    detailsOFF.to_csv(os.path.join(wdir +"/data/detailsOFF.csv"), mode='w', header=True)

else:
    list_ch = ['ACCx', 'ACCy', 'ACCz', 'GYRx', 'GYRy', 'GYRz']
    for idx, n in enumerate(list_ch):
        if idx == 0:
            datON = pds.read_csv(os.path.join(wdir+"/data/dataON_" +n +".csv"), sep=';')
            datOFF = pds.read_csv(os.path.join(wdir + "/data/dataOFF_" + n + ".csv"), sep=';')
        else:
            datON = np.dstack((datON, pds.read_csv(os.path.join(wdir+"/data/dataON_" +n +".csv"), sep=';')))
            datOFF = np.dstack((datOFF, pds.read_csv(os.path.join(wdir+"/data/dataOFF_" +n +".csv"), sep=';')))

    detailsON = pds.read_csv(os.path.join(wdir +"/data/detailsON.csv"))
    detailsON = pds.read_csv(os.path.join(wdir +"/data/detailsOFF.csv"))

print(datON.shape)

# start categorising data in order to prepare the cnn model training/estimation
catobj = preprocess.dataworks.Categorize()
smplsONtrain, smplsONtest, smplsOFFtrain, smplsOFFtest = catobj.subsample_data(datON, datOFF, .8)
# TODO at this point a permutation for e.g. k-fold crossvalidation, or permutation approaches could be included here
trainX, trainy = catobj.create_cat(smplsONtrain, smplsOFFtrain)
testX, testy = catobj.create_cat(smplsONtest, smplsOFFtest)

tune_params = False # tuning the hyperparameters doesn't work so far for unknon reasons. Supposedly, 3D data is not recognised properly!! Don't use
if tune_params == True:

    p = {'lr': (0.5, 5, 10),
              'first_neuron': [4, 8, 16, 32, 64],
              'hidden_layers': [0, 1, 2],
              'batch_size': (2, 30, 10),
              'epochs': [150],
              'dropout': (0, 0.5, 5),
              'weight_regulizer': [None],
              'emb_output_dims': [None],
              'shape': ['brick', 'long_funnel'],
              'optimizer': ['Adam'], #[Adam, Nadam, RMSprop],
              'losses': ['logcosh'], #[logcosh, binary_crossentropy],
              #'activation': ['relu'],#[relu, elu],
              'last_activation': ['sigmoid']}


    def evaluate_params(trainX, trainy, testX, testy, params):
        # next we can build the model exactly like we would normally do it
        model = Sequential()
        #model.add(Conv1D(filters=32, kernel_size=7, activation='relu'))
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

        out, model = evaluate_params(trainX, trainy, testX, testy, params)

    scan_object = ta.Scan(trainX, trainy, model=evaluate_params, params=p, experiment_name='basic_model')

else:
    # start evaluating and estimation cnn models
    cnnobj = cnn.estimate_cnn.ModelDefinition(trainX, trainy, testX, testy)

    repeats = 10
    n_filter = [32, 64]#[32, 64, 128]
    #n_kernel = [7, 9, 11, 13]
    n_kernel = [25,10]#[50, 25, 10, 5]
    scores = list()
    f_scores = list()
    k_scores = list()

    for k in n_kernel:
        for f in n_filter:
            for r in range(repeats):
                #score = cnnobj.evaluate_combined_model(trainX, trainy, testX, testy, f, k)
                score = cnnobj.evaluate_model(trainX, trainy, testX, testy, f, k)

                score = score * 100.0
                print('>F=%d; K=%d; #%d: %.3f' % (f, k, r + 1, score))
            scores.append(score)
        f_scores.append(scores)
    k_scores.append(f_scores)