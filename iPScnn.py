#!/usr/bin/env python
# -*- coding: utf-8 -*-

import preprocess.dataworks
import cnn.estimate_cnn
from importlib import reload

reload(cnn.estimate_cnn)

# loads and preprocesses data as needed
filepath = '/media/david/windows/Dokumente und Einstellungen/dpedr/Jottacloud/onoff_svm'
datobj = preprocess.dataworks.DataPipeline(filepath, 'all', 'all','all',ignbad=True)
datobj.generate_subjlist()
datON, datOFF = datobj.load_all()

# start categorising data in order to prepare the cnn model training/estimation
catobj = preprocess.dataworks.Categorize()
smplsONtrain, smplsONtest, smplsOFFtrain, smplsOFFtest = catobj.subsample_data(datON, datOFF, .35)
# TODO at this point a permutation for e.g. k-fold crossvalidation, or permutation approaches could be included here
trainX, trainy = catobj.create_cat(smplsONtrain, smplsOFFtrain)
testX, testy = catobj.create_cat(smplsONtest, smplsOFFtest)

# start evaluating and estimation cnn models
cnnobj = cnn.estimate_cnn.ModelDefinition(trainX, trainy, testX, testy)

repeats = 2
n_filter = [32, 64]
n_kernel = [3, 5]
scores = list()
f_scores = list()
k_scores = list()

for k in n_kernel:
    for f in n_filter:
        for r in range(repeats):
            score = cnnobj.evaluate_model(trainX, trainy, testX, testy, f, k)
            score = score * 100.0
            print('>F=%d; K=%d; #%d: %.3f' % (f, k, r + 1, score))
        scores.append(score)
    f_scores.append(scores)
k_scores.append(f_scores)

