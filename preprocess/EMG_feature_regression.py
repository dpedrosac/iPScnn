#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, sys
import preprocess.EMGprocess
import preprocess.plot_features
from typing import List, Dict, Sized
from sklearn.metrics import confusion_matrix
import pickle
import pandas as pds
import numpy as np
from time import sleep
import glob
import pickle

#  ------- ------- ------- ------- General settings ------- ------- ------- ------- ------- ------- ------- -------
conds = ['OFF', 'ON']               # conditions to test for
COI = [4,5,8]                       # channels of interest
task = 'tap'                        # selected task
n_splits = 10                       # n-fold cross validation
features = ["IAV", "MAV", "RMS",    # all features used for the different (multiple) regressions
            "WAMP", "WL",
            "ZC", "SSC", "VAR"]

fileobj = preprocess.EMGprocess.EMGfileworks(task=task, subj='all', scaling=False)
svmobj = preprocess.EMGprocess.EMGpredict()
EMGdetails = preprocess.EMGprocess.ExtractEMGfeat(task=task, filename_append='')
details = EMGdetails.extract_details(subj=fileobj.subj, act="return")
#  ------- ------- ------- ------- ------- ------- ------- ------- ------- ------- ------- ------- ------- -------

# Extracts available recordings for <task> and for all subj in <fileobj>; filters/extracts features (if not done yet)
listEMGallOFF, listEMGallON = fileobj.get_EMGfilelist_all(task=task)
list_type = {"OFF" : listEMGallOFF, "ON": listEMGallON}

dfs_file = os.path.join(os.path.join(fileobj.datobj.wdir, 'data', 'EMG'), "dfs_pickled.pkl")
if not os.path.isfile(os.path.join(fileobj.datobj.wdir, 'data', 'EMG', dfs_file)):
    # Load EMG features to memory, according to file-list <listEMGallOFF> (here only 8secs. are used)
    progressbar_size = len(listEMGallOFF + listEMGallON)
    print('Reading extracted features for all subjects ...', end='', flush=True)
    dfs: Dict = {}
    for idx, r in enumerate(listEMGallOFF + listEMGallON):
        j = (idx + 1) / progressbar_size
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j))
        sys.stdout.flush()
        sleep(0.25)

        filename = os.path.splitext(r)[0]
        infile = open(os.path.join(fileobj.datobj.wdir, 'data', 'EMG', 'features_split8', filename + '_8secs_features.pkl'),
                      'rb')
        dfs[r] = pickle.load(infile)
        infile.close()
        idx_subj = details.index[details['Name'] == filename[0:11]].tolist()
        if "OFF" in filename:
            dfs[r].insert(0, "output_0", 0)
            dfs[r].insert(1, "UPDRS", list(details["updrsOFF"].iloc[idx_subj])*len(dfs[r]))
        else:
            dfs[r].insert(0, "output_0", 1)
            dfs[r].insert(1, "UPDRS", list(details["updrsON"].iloc[idx_subj])*len(dfs[r]))

    with open(dfs_file, 'wb') as handle:
        #pickle.dump(dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(dfs, handle, protocol=2)
else:
    with open(dfs_file, 'rb') as handle:
        dfs = pickle.load(handle)
print("DONE!")
print()

# Extract all features of interest for both conditions
column_regex = re.compile("^((" + ")|(".join(features) + "))_[0-9]+")
feat_on = pds.DataFrame(columns=features)
feat_off = pds.DataFrame(columns=features)

for c in conds:
    feat_all = {}
    print()
    list_temp = fileobj.get_EMGfilelist_per_subj(cond=c, task=task, subj='')
    progressbar_size = len(list_temp)
    print("Reading " + c + " condition features per subject")

    for idx, r in enumerate(list_temp):
        j = (idx + 1) / progressbar_size
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j))
        sys.stdout.flush()
        sleep(0.25)

        data_temp = pds.DataFrame()
        for f in list_temp[r]:
            filename = os.path.splitext(f)[0]
            columns_input = list(filter(column_regex.match, list(dfs[f])))
            data_temp = pds.concat([data_temp, dfs[f][columns_input]])

        feat_all[idx + 1] = pds.DataFrame.mean(data_temp, skipna=True)

    if c == 'OFF':
        featOFF_Allchan = pds.DataFrame.from_dict(feat_all, orient='index')
    else:
        featON_Allchan = pds.DataFrame.from_dict(feat_all, orient='index')
print()

# Select only features from channels of interest <COI> defined above
regex = re.compile(r'[A-Z]+_[0-9]+')
cols = list(filter(regex.search, list(featOFF_Allchan.columns.values)))
regex = re.compile(r'[A-Z]+[0-9]+_[0-9]+')
cols = cols + list(filter(regex.search, list(featOFF_Allchan.columns.values)))
column_regex = re.compile("^((" + ")|(".join(features) + "))_[0-9]+")

for n in features:
    temp = [s for s in cols if n in s]
    cols_sel = list()
    for k in COI:
        temp_feature = [s for s in temp if str(k) in s[-1]]
        cols_sel = cols_sel + temp_feature
    feat_on[n] = pds.DataFrame.mean(featON_Allchan[cols_sel], axis=1, skipna=True)
    feat_off[n] = pds.DataFrame.mean(featOFF_Allchan[cols_sel], axis=1, skipna=True)

# Extract the relative features and add the changes in UPDRS to start the regression analyses
feat_regr = pds.DataFrame(columns=features)

# Estimate UPDRS change and changes in features for later running multiple regressions
list_temp = fileobj.get_EMGfilelist_per_subj(cond='ON', task=task, subj='')

print()
print('Preparing data for regression ...', end='', flush=True)
dfreg_file = os.path.join(os.path.join(fileobj.datobj.wdir, 'data', 'EMG'), "dfreg_pickled.pkl")
if not os.path.isfile(os.path.join(fileobj.datobj.wdir, 'data', 'EMG', dfreg_file)):
    dfreg: Dict = {}
    progressbar_size = len(listEMGallON)*len(features)
    iter = 0

    dfreg: Dict = {}
    for idx, r in enumerate(list_temp):
        idx_subj = details.index[details["Name"] == r]

        for f in list_temp[r]:
            data_temp = pds.DataFrame(columns=features)
            for n in features:
                iter = iter + 1
                j = iter / progressbar_size
                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j))
                sys.stdout.flush()
                sleep(0.1)

                data_temp[n] = np.mean(dfs[f][cols_sel].values.flatten()) - feat_off[n][idx_subj + 1]
                data_temp[n].index = np.arange(0, len(data_temp[n].index))
                updrs_temp = 100 * np.divide(details.iloc[idx_subj]["updrsOFF"] - details.iloc[idx_subj]["updrsON"],
                                                     details.iloc[idx_subj]["updrsOFF"])  # details["updrsDiff"]

            data_temp.insert(0, "output_0", value=float(updrs_temp))
            dfreg[f] = data_temp

    with open(dfreg_file, 'wb') as handle:
        pickle.dump(dfreg, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(dfreg_file, 'rb') as handle:
        dfreg = pickle.load(handle)
print("DONE!")
print()

# Start "shallow learning" routine for the EMG feature data
feature_sets = {
    "RMS": ["RMS"],
    "Hudgins": ["MAV", "WL", "ZC", "SSC"],
    "Du": ["IAV", "VAR", "WL", "ZC", "SSC", "WAMP"],
    "all": ["IAV", "MAV", "VAR", "WL", "ZC", "SSC", "WAMP", "RMS"]
}

# Subtypes of patients to be discriminated
types = {
    0: "OFF",
    1: "ON"
}

# Classifiers and its options to be used for the distinct classifiers
classifiers = {
    "LR":
        {"predictor": "LinearRegression",
         "args": {"fit_intercept": True, "normalize": "False"}},

    "LassoRegression": {
        "predictor": "Lasso",
        "args": {"alpha": 1.0, "fit_intercept": True, "max_iter": 10000, "selection": "cyclic"}},

    "SVR_rbf":
        {"predictor": "SVR",
         "args": {"C": 50.0, "kernel": "rbf", "degree": 3, "gamma": "auto", "coef0": 0.0,
                  "shrinking": True, "tol": 0.001, "cache_size": 200, "verbose": False}},

    "SVR_poly":
         {"predictor": "SVR",
          "args": {"C": 50.0, "kernel": "poly", "degree": 3, "gamma": "auto", "coef0": 0.0,
                  "shrinking": True, "tol": 0.001, "cache_size": 200, "verbose": False}},

    "kNNRegression": {
        "predictor": "kNNRegression",
        "args": {"n_neighbors": 5, "weights": "uniform", "algorithm": "auto", "leaf_size": 30, "p": 2,
                 "metric": "minkowski", "metric_params": None, "n_jobs": None}}
}

# Start splitting data in order to perform k-fold cross-validation
splits_type = fileobj.split_per_id(list_type, n_splits=n_splits, test_size=.2)
splits_all = {}
for n in range(int(n_splits)):
    splits_temp1 = splits_type[1][n]["train"]
    splits_temp2 = splits_type[1][n]["test"]
    splits_all.update({n: {"train": splits_temp1, "test": splits_temp2}})

# Start shallow learn classification
print()
print('Starting the classifiers')

output: Dict[str, any] = dict()
output["classifiers"] = classifiers
output["feature_sets"] = feature_sets
output["results"]: List[Dict[str, any]] = list()

# Start splitting data and running analyses
for id_, id_splits in splits_all.items():  # k-fold-validation

    for feature_set_name, features in feature_sets.items():  # features to compare
        print('\t\tFeature set: {:s} -'.format(feature_set_name), end='', flush=True)

        # prepare data, i.e. isolate features of interest according to <features>, rename -> input_XXX and create  index
        data = fileobj.prepare_data_regression(dfreg, id_splits, features)

        # list columns containing only feature data
        regex = re.compile(r'input_[0-9]+_[A-Z]+')
        cols = list(filter(regex.search, list(data["train"].columns.values)))

        # Extract limited training x and y, only with chosen channel configuration
        train_x = data["train"][cols]
        train_y = data["train"]["output_0"]

        # Extract limited testing x and y, only with chosen channel configuration
        test_x = data["test"][cols]
        test_y_true = data["test"]["output_0"]

        for clf_id, clf_settings in classifiers.items():  # loop through different classifiers
            print(' {:s}'.format(clf_id), end='', flush=True)

            # Prepare classifier pipeline and fit the classifier to train data
            pipeline = svmobj.prepare_pipeline(train_x, train_y,
                                               predictor=clf_settings["predictor"],
                                               norm_per_feature=False,
                                               **clf_settings["args"])

            # Run prediction on test data
            test_y_pred = pipeline.predict(test_x)
            train_y_pred = pipeline.predict(train_x)

            # save classification results to output structure
            output["results"].append({"id": id_, "clf": clf_id,
                                      "feature_set": feature_set_name,
                                      "y_true": test_y_true.values.astype(int), "y_pred": test_y_pred,
                                      "ytrain_true": train_y.values.astype(int), "ytrain_pred": train_y_pred,
                                      "trainX": train_x, "testX": test_x,
                                      "best_params": pipeline.best_params_})
        print()

    # Save all data to separate file, which may be used later for plotting purposes
pickle.dump(output, open(
    os.path.join(fileobj.datobj.wdir, "data", "EMG", "results", "results_shallow_regression.bin"), "wb"),protocol=2)
