#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re
import preprocess.EMGprocess
from typing import List, Dict, Sized
from sklearn.metrics import confusion_matrix
import pickle

## General part in which, settings are defined and general classes/functions are defined for later
#conds = ['OFF', 'ON']  # conditions to test for
COI = [1,2,3,4,5,6,7,8]  # channel of interest
fileobj = preprocess.EMGprocess.EMGfileworks('rst', subj='all', scaling=False)
svmobj = preprocess.EMGprocess.EMGpredict()
EMGdetails = preprocess.EMGprocess.ExtractEMGfeat(task='rst', filename_append='')
details = EMGdetails.extract_details(subj=fileobj.subj, act="return")

#for c in conds:
#    if c == 'OFF':
#        listEMGsubjOFF = fileobj.get_EMGfilelist_per_subj(cond=c, task='rst', subj='')
#    else:
#        listEMGsubjON = fileobj.get_EMGfilelist_per_subj(cond=c, task='rst', subj='')

# Get a list of all recordings available for the "rst"-condition and for all subjects listed in fileobj, filter and
# extract features (if not already present)
listEMGallOFF, listEMGallON = fileobj.get_EMGfilelist_all(task='rst')
fileobj.filter_data_and_extract_features(listEMGallOFF, '', '', '')
fileobj.filter_data_and_extract_features(listEMGallON, '', '', '')

# Put data into correct format
trem_details = details[details.type == 1]
bradkin_details = details[(details.type == 0) | (details.type == 2)]

# Sort data into two categories: a) bradikentic-rigid and b) tremordominant iPS-patients
list_type = {}
tremdom = []
brakin = []
for r in listEMGallOFF:
    filename = os.path.splitext(r)[0]
    if any(bradkin_details["Name"].str.find(filename[0:11]) == 0):
        brakin.append(r)
    else:
        tremdom.append(r)
list_type.update({"brakin": brakin, "tremdom": tremdom})

# listEMGtremOFF = fileobj.get_EMGfilelist_per_subj(cond='OFF', task='rst', subj=list(trem_details.Name))
# listEMGbrakinOFF = fileobj.get_EMGfilelist_per_subj(cond='OFF', task='rst', subj=list(bradkin_details.Name))

# Load feature data to memory according to the list if files from before
dfs: Dict = {}
for r in listEMGallOFF:
    print("Reading features for input file: ", r)
    filename = os.path.splitext(r)[0]
    infile = open(os.path.join(fileobj.datobj.wdir, 'data', 'EMG', 'features_split', filename + '_features.pkl'), 'rb')
    dfs[r] = pickle.load(infile)
    infile.close()
    if any(bradkin_details["Name"].str.find(filename[0:11]) == 0):
        dfs[r].insert(0, "output_0", 0)
    else:
        dfs[r].insert(0, "output_0", 1)

# Start splitting data in order to perform k-fold cross-validation
n_splits = 5
splits_type = fileobj.split_per_id(list_type, n_splits=n_splits, test_size=.2)

splits_all = {}
splits_temp = []
splits2 = []
for n in range(int(n_splits)):
    splits_temp1 = splits_type[0][n]["train"] + splits_type[1][n]["train"]
    splits_temp2 = splits_type[0][n]["test"] + splits_type[1][n]["test"]
    splits_all.update({n: {"train": splits_temp1, "test": splits_temp2}})

# Start shallow learn classification
print()
print('Starting the classifiers')

# Feature sets to be used for the discrimination between different subtypes
feature_sets = {
    "RMS": ["RMS"],
    "Hudgins": ["MAV", "WL", "ZC", "SSC"],
    "Du": ["IAV", "VAR", "WL", "ZC", "SSC", "WAMP"]
}

# Subtypes of patients to be discriminated
types = {
    0: "bradykinetic-rigid iPS",
    1: "tremordominant iPS"
}

# Classifiers and its options to be used for the distinct classifiers
classifiers = {
    "LDA": {
        "predictor": "LDA",
        "args": {"solver": "svd", "shrinkage": None, "priors": None, "n_components": None,
                 "store_covariance": False, "tol": 0.0001}},
    "QDA": {
        "predictor": "QDA",
        "args": {"priors": None, "reg_param": 0.3, "store_covariance": False, "tol": 0.0001}},
    "kNN": {
        "predictor": "kNN",
        "args": {"n_neighbors": 5, "weights": "uniform", "algorithm": "auto", "leaf_size": 30, "p": 2,
                 "metric": "minkowski", "metric_params": None, "n_jobs": None}},
    "SVM":
        {"predictor": "SVM",
         "args": {"C": 50.0, "kernel": "rbf", "degree": 3, "gamma": "auto", "coef0": 0.0,
                  "shrinking": True, "probability": False, "tol": 0.001, "cache_size": 200,
                  "class_weight": None, "verbose": False, "max_iter": -1, "decision_function_shape": "ovr",
                  "random_state": None}}
}

output: Dict[str, any] = dict()
output["types"] = types
output["classifiers"] = classifiers
output["feature_sets"] = feature_sets
output["results"]: List[Dict[str, any]] = list()

# Start splitting data and running analyses
for id_, id_splits in splits_all.items(): # k-fold-validation

    for feature_set_name, features in feature_sets.items(): # features to compare
        print('\t\tFeature set: {:s} -'.format(feature_set_name), end='', flush=True)

        # prepare data, i.e. isolate features of interest accordint to <features>, rename -> input_XXX and create  index
        data = fileobj.prepare_data_complete(dfs, id_splits, features)

        # list columns containing only feature data
        regex = re.compile(r'input_[0-9]+_[A-Z]+_[0-9]+')
        cols = list(filter(regex.search, list(data["train"].columns.values)))

        cols_sel = list()
        for k in COI:
            temp = [s for s in cols if str(k) in s[-1]]
            cols_sel = cols_sel + temp

        # Extract limited training x and y, only with chosen channel configuration
        train_x = data["train"][cols_sel]
        train_y = data["train"]["output_0"]

        # Extract limited testing x and y, only with chosen channel configuration
        test_x = data["test"][cols_sel]
        test_y_true = data["test"]["output_0"]

        for clf_id, clf_settings in classifiers.items(): # loop through different classifiers
            print(' {:s}'.format(clf_id), end='', flush=True)

            # Prepare classifier pipeline and fit the classifier to train data
            pipeline = svmobj.prepare_pipeline(train_x, train_y,
                                               predictor=clf_settings["predictor"],
                                               norm_per_feature=False,
                                               **clf_settings["args"])

            # Run prediction on test data
            test_y_pred = pipeline.predict(test_x)

            # Calculate confusion matrix
            cm = confusion_matrix(test_y_true.values.astype(int), test_y_pred, list(types.keys()))

            # save classification results to output structure
            output["results"].append({"id": id_, "clf": clf_id,
                                      "feature_set": feature_set_name,
                                      "cm": cm, "y_true": test_y_true.values.astype(int),
                                      "y_pred": test_y_pred})
        print()

# Save all data to separate file, which may be used later for plotting purposes
pickle.dump(output, open(os.path.join(fileobj.datobj.wdir, "data", "EMG", "results", "results_tremorclass.bin"), "wb"))
