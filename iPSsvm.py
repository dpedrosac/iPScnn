#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import libraries
import os
# reduce verbosity by disabling KMP affinity messages
os.environ['KMP_AFFINITY'] = 'noverbose'
import preprocess.dataworks
import numpy as np
import pandas as pds
from tsfresh import extract_features
from sklearn import svm

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
# do feature extraction for training data
firstrun = True
for i in range(0, trainX.shape[0]):
    print("Train data trial ", i, " of ", trainX.shape[0], ", ", i / trainX.shape[0] * 100, " percent complete.")
    curr_data = trainX[i,:,:]
   # append time index as first column (will be second column after next step)
    curr_data = np.append(np.reshape(time, (curr_data.shape[0], 1)), curr_data, axis = 1)
   # append trial index as first column (currently always one, but see below TODO)
    curr_data = np.append(np.ones((curr_data.shape[0],1)), curr_data, axis = 1)
   # convert to pandas dataframe
    df = pds.DataFrame(curr_data)
   # extract time series features
    ef = extract_features(df, column_id = 0, column_sort = 1)
    if firstrun:
       # define output as beeing the first row of extracted features from first trial
        features_trainX = ef
        firstrun = False
    else:
       # append features from the second trial on to existing matrix
        features_trainX = np.append(features_trainX, ef, axis = 0)

# do feature extraction for test data
firstrun = True
for i in range(0, testX.shape[0]):
    print("Test data trial ", i, " of ", testX.shape[0], ", ", i / testX.shape[0] * 100, " percent complete.")
    curr_data = testX[i,:,:]
   # append time index as first column (will be second column after next step)
    curr_data = np.append(np.reshape(time, (curr_data.shape[0], 1)), curr_data, axis = 1)
   # append trial index as first column (currently always one, but see below TODO)
    curr_data = np.append(np.ones((curr_data.shape[0],1)), curr_data, axis = 1)
   # convert to pandas dataframe
    df = pds.DataFrame(curr_data)
   # extract time series features
    ef = extract_features(df, column_id = 0, column_sort = 1)
    if firstrun:
       # define output as beeing the first row of extracted features from first trial
        features_testX = ef
        firstrun = False
    else:
       # append features from the second trial on to existing matrix
        features_testX = np.append(features_testX, ef, axis = 0)

# do support vector classification
clf = svm.SVC(gamma = 'scale')
clf.fit(features_trainX, trainy[:,0])
t = clf.predict(features_testX)
t2 = testy[:,0]
print(pds.crosstab(t, t2, margins = True))


# INFO: run script from console like so:
# exec(open('iPSsvm.py').read())
# TODO:
# currently feature extraction is implemented trial-wise. In the future feature extraction
# should be done across the whole sample in order to enable consistent feature reduction afterwards.
# for this, probably there has to be a trial index (which is currently implemented by ones) and
# input data for extract_features have to be arranged as a long table with
# rows = ntrials * length of input data
# cols = trial index | time index | data (1-n) ...
# TODO:
# make feature extraction a function instead of two loops
