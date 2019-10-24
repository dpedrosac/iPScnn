#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import libraries
import preprocess.dataworks
import numpy as np
import pandas as pds
from tsfresh import extract_features

#if False:
if True:
    # loads and preprocesses data as needed
    datobj = preprocess.dataworks.DataPipeline('', 'all', 'all', 'all',ignbad=True)
    datobj.generate_subjlist()
    datON, datOFF, detON, detOFF = datobj.load_all(window = False)
    catobj = preprocess.dataworks.Categorize()
    smplsONtrain, smplsONtest, smplsOFFtrain, smplsOFFtest = catobj.subsample_data(datON, datOFF, .25)
    trainX, trainy = catobj.create_cat(smplsONtrain, smplsOFFtrain)
    testX, testy = catobj.create_cat(smplsONtest, smplsOFFtest)

time = np.arange(trainX[0,:,:].shape[0])
firstrun = True
for i in range(0, trainX.shape[0]):
    curr_data = trainX[i,:,:]
    curr_data = np.append(np.reshape(time, (curr_data.shape[0], 1)), curr_data, axis = 1)
    curr_data = np.append(np.ones((curr_data.shape[0],1)), curr_data, axis = 1)
    df = pds.DataFrame(curr_data)
    ef = extract_features(df, column_id = 0, column_sort = 1)
    if firstrun:
        X = ef
        firstrun = False
    else:
        X = np.append(X, ef, axis = 0)


# run script from console like this:
# exec(open('iPSsvm.py').read())

