#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import libraries
import preprocess.dataworks
import numpy as np
import pandas as pds
from tsfresh import extract_features

if False:
    # loads and preprocesses data as needed
    datobj = preprocess.dataworks.DataPipeline('', 'all', 'all', 'all',ignbad=True)
    datobj.generate_subjlist()
    datON, datOFF = datobj.load_all(window = False)
    catobj = preprocess.dataworks.Categorize()
    smplsONtrain, smplsONtest, smplsOFFtrain, smplsOFFtest = catobj.subsample_data(datON, datOFF, .25)
    trainX, trainy = catobj.create_cat(smplsONtrain, smplsOFFtrain)
    testX, testy = catobj.create_cat(smplsONtest, smplsOFFtest)

a = trainX[0,:,:]
time = np.arange(a.shape[0])
b = np.append(np.reshape(time,(a.shape[0],1)),a,axis = 1)
c = np.append(np.ones((a.shape[0],1)),b,axis = 1)
df = pds.DataFrame(data = c[:,2:], index = c[:,1])