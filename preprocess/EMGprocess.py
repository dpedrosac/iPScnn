#!/usr/bin/env python
# -*- coding: utf-8 -*-

import preprocess.EMGfeatures
import preprocess.EMGfilter

from typing import List, Dict, Sized

import os, glob, warnings, math, ast, re
import preprocess.dataworks
from scipy.special import comb
import numpy as np
import pandas as pds
import matplotlib.pyplot as plt
import pickle
import getpass, yaml
from numpy.lib.stride_tricks import as_strided
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

class EMGfileworks:
    """The next lines are a set of functions intended to read filenames and return for further analyses"""

    def __init__(self, task, subj='all', scaling=False):

        self.debug = False
        self.samplerate = 200
        self.device = 'EMG'
        self.task = task
        self.datobj = preprocess.dataworks.DataPipeline('', self.task, self.device, '')
        if subj == 'all':
            self.subj = self.datobj.subjlist
        else:
            self.subj = subj

        self.scaling = scaling
        if getpass.getuser() == "urs":
            with open('/home/urs/sync/projects/autostim/analysis/iPScnn/config.yaml', 'r') as f:
                d = yaml.load(f.read())
        elif getpass.getuser() == "dpedr":
            with open('D:/iPScnn/config.yaml', 'r') as f:
                d = yaml.load(f.read())
        else:
            with open('/media/storage/iPScnn/config.yaml', 'r') as f:
                d = yaml.load(f.read())
        self.emgdir = d[0]["dataworks"]['folders'][getpass.getuser()]['emgrawdir']
        self.filt = preprocess.EMGfilter.Filter()

    def file_browser(self, word):
        """function which helps to find files with certain content, specified in the input"""
        file = []
        os.chdir(self.emgdir)

        for f in glob.glob("*"):
            if word in f:
                file.append(f)

        return file

    def get_EMGfilelist_all(self, task=''):
        """this function returns the different EMG files, that should be extracted and processed later. The reason
                for this approach instead of loading data strainght into the program is, that this way both k-fold
                validation and splitting into train/test datasets is easier"""

        conds = ['ON', 'OFF']
        emg_filesON = []
        emg_filesOFF = []
        print("Extracting all filenames for subjects included in the study", flush=True)
        for idx, pseud in enumerate(self.subj):
            for c in conds:
                filename = pseud + "_" + task + "_" + c + "_EMG"
                all_files = self.file_browser(filename)

                if c == 'ON':
                    emg_filesON.append(all_files)
                else:
                    emg_filesOFF.append(all_files)

        return list(itertools.chain(*emg_filesOFF)), list(itertools.chain(*emg_filesON))

    def get_EMGfilelist_per_subj(self, cond='unknown', task='unknown', subj=list()):
        """this function returns the different EMG files subsampled for every subject analysed and according to a
        condition and a task that needs to be specified"""

        if (cond != 'ON') and (cond != 'OFF'):
            raise ValueError(cond + ' is not a valid definition of a condition, please use "ON" or "OFF" instead')

        if (task != 'rst') and (task != 'hld') and (task != 'tap') and (task != 'dd'):
            raise ValueError(task + ' is not a valid definition of a tas, please use "rst", "tap", "dd" or "hld" instead')

        if not subj:
            subj = self.subj

        print("Extracting filenames per subjects included in the study", flush=True)
        emg_filesSUBJ = {}
        for idx, pseud in enumerate(subj):
            filename = pseud + "_" + task + "_" + cond + "_EMG"
            all_files = self.file_browser(filename)
            emg_filesSUBJ[pseud] = all_files

        return emg_filesSUBJ

    def load_file(self, filename):
        """ helper function that reads data and returns the values into a dataframe; there are two options:
        1)  True: Mean = 0 and Std = 1
        2)  False: Mean = 0, Std remains """
        from sklearn import preprocessing

        # read data from txt-file as processed via MATLAB
        df = pds.read_table(filename, header=None, sep='\s+')

        if self.scaling:
            return preprocessing.robust_scale(df.values)
            # return preprocessing.normalize(df.values)
        else:
            return df.values

    def filter_data_and_extract_features(self, list_files, data_folder, filtered_data_folder, feature_folder, dur):
        """this functions filters data in order to proceed with feature extraction later"""
        filtobj = preprocess.EMGfilter.Filter
        file_features = self.datobj.wdir + "/preprocess/all_features.xml"

        if data_folder == '':
            data_folder = self.emgdir

        if dur == 'tot':
            dur = 8

        if filtered_data_folder == '':
            filtered_data_folder = os.path.join(self.datobj.wdir, 'data', 'EMG', 'filtered_data')

        if feature_folder == '':
            feature_folder = os.path.join(self.datobj.wdir, 'data', 'EMG', 'features_split')

        if not os.path.exists(filtered_data_folder):
            os.makedirs(filtered_data_folder)

        if not os.path.exists(feature_folder):
            os.makedirs(feature_folder)

        # by each filename in download folder
        for file in list_files:
            output_file = os.path.splitext(file)[0] + "_" + str(dur) + 'secs_filtered.csv'
            output_file2 = os.path.splitext(file)[0] + "_" + str(dur) + 'secs_features.pkl'
            if not (os.path.isfile(os.path.join(filtered_data_folder, output_file)) and
                    os.path.isfile(os.path.join(feature_folder, output_file2))):
                datemg = list()
                dattemp = self.load_file(os.path.join(data_folder, file))
                datemg.append(dattemp[0:dur*200])

                emg = pds.DataFrame(np.vstack(datemg), columns=["EMG_1", "EMG_2", "EMG_3", "EMG_4",
                                                                "EMG_5", "EMG_6", "EMG_7", "EMG_8"])
                if self.debug:
                    import scipy.signal as sgn
                    fig, ax = plt.subplots(2)
                    ax[0].plot(emg["EMG_8"])
                    Psd, f = sgn.welch(emg["EMG_8"], 200, nperseg=256)
                    ax[1].plot(Psd, f)
                    plt.show()

                self.filt.apply_filter(emg)

                if self.debug:
                    ax[0].plot(emg["EMG_8"])
                    Psd, f = sgn.welch(emg["EMG_8"], 200, nperseg=256)
                    ax[1].plot(Psd, f)
                    plt.show()

                print('Saving to file: {:s}'.format(output_file))
                emg.to_csv(os.path.join(filtered_data_folder, output_file), mode='w', header=True)

                # Extract all available features
                df = preprocess.EMGfeatures.features_from_xml_on_df(file_features, emg)
                df.to_pickle(os.path.join(feature_folder, output_file2))
            elif (os.path.isfile(os.path.join(filtered_data_folder, output_file))) & ("trial1_" in file):
                print("Filtering and feature extraction for " + file[0:11] + " at " + str(dur) +
                      " sec. duration already finished, continuing ...")
            else:
                continue

    def split(self, items: Sized, n_splits=None, test_size=0.1, train_size=None, random_state=None):
        """split function which ensures that data is split into k number of chunks in order to run k-fold
        validation in the routine later"""
        rng1 = np.random.RandomState(random_state)

        if test_size is None and train_size is None:
            raise ValueError("Missing test size or train size")

        data_count = len(items)
        items = set(range(data_count))

        if isinstance(train_size, float):
            train_size = np.rint(train_size * data_count)
        if isinstance(test_size, float):
            test_size = np.rint(test_size * data_count)
        if train_size is None:
            train_size = data_count - test_size
        if test_size is None:
            test_size = data_count - train_size

        train_size = int(train_size)
        test_size = int(test_size)

        if train_size < 1 or train_size > (data_count - 1):
            raise ValueError("Wrong train size: train_size={:d},test_size={:d} out of {:d}".
                             format(train_size, test_size, data_count))
        if test_size < 1 or test_size > (data_count - 1):
            raise ValueError("Wrong test size: train_size={:d},test_size={:d} out of {:d}".
                             format(train_size, test_size, data_count))

        n_comb = int(comb(data_count, train_size) * comb(data_count - train_size, test_size))

        if n_splits is None:
            n_splits = n_comb
        if n_splits > n_comb:
            warnings.warn("n_splits larger than available ({:d}/{:d})".format(n_splits, n_comb))
            n_splits = n_comb

        splits = []
        while len(splits) < n_splits:
            items_train = rng1.choice(list(items), size=train_size, replace=False)
            items_left = items.copy()
            for it in items_train:
                items_left.remove(it)
            items_test = rng1.choice(list(items_left), size=test_size, replace=False)
            split_candidate = (set(items_train), set(items_test))
            if split_candidate not in splits:
                splits.append(split_candidate)

        return splits

    def split_per_id(self, records, n_splits: int = None, test_size=0.2):
        """splits data to train and test datasets according to the settings"""

        print("Creating two datasets: train and test data for this group", flush=True)
        sets = {}
        for idx, pseud in enumerate(records):
            rec_temp = records[pseud]
            splits = self.split(rec_temp, n_splits=n_splits, test_size=test_size, random_state=0)

            record_splits = []
            for train_index, test_index in splits:
                train_records = [rec_temp[idx2] for idx2 in train_index]
                test_records = [rec_temp[idx2] for idx2 in test_index]
                record_splits.append({"train": train_records, "test": test_records})

            sets[idx] = record_splits

        return sets

    def prepare_data_complete(self, dfs, s, features):
        metadata = ['output_0']

        dfs_output: Dict[str, pds.DataFrame] = dict()
        column_regex = re.compile("^((" + ")|(".join(features) + "))_[0-9]+")

        for k, v in s.items():
            df_temp = pds.DataFrame()
            columns_input = []
            for r in v:
                columns_input = list(filter(column_regex.match, list(dfs[r])))
                df_temp = df_temp.append(dfs[r][columns_input + metadata])

            df_temp.rename({c: "input_{:d}_{:s}".format(i, c) for i, c in enumerate(columns_input)},
                           axis="columns", inplace=True)

            dfs_output[k] = df_temp
            dfs_output[k].index = np.arange(0, len(dfs_output[k].index))
        return dfs_output


    def prepare_data_regression(self, dfreg, s, features):
        metadata = ['output_0']

        dfs_output: Dict[str, pds.DataFrame] = dict()
        column_regex = re.compile("^((" + ")|(".join(features) + "))+")

        for k, v in s.items():
            df_temp = pds.DataFrame()
            columns_input = []
            for r in v:
                columns_input = list(filter(column_regex.match, list(dfreg[r])))
                df_temp = df_temp.append(dfreg[r][columns_input + metadata])

            df_temp.rename({c: "input_{:d}_{:s}".format(i, c) for i, c in enumerate(columns_input)},
                           axis="columns", inplace=True)

            dfs_output[k] = df_temp
            dfs_output[k].index = np.arange(0, len(dfs_output[k].index))
        return dfs_output

class ExtractEMGfeat:
    """functions required to etxract EMG data in order to process it later"""

    def __init__(self, task, filename_append):
        self.debug = False
        self.device = 'EMG'
        self.task = task
        self.datobj = preprocess.dataworks.DataPipeline('', self.task, self.device, '')
        self.filt = preprocess.EMGfilter.Filter()

        if filename_append == '':
            self.file_trunk = self.datobj.wdir + "/data/EMG/features_subj/"
        else:
            self.file_trunk = self.datobj.wdir + "/data/EMG/" + filename_append

        if getpass.getuser() == "urs":
            with open('/home/urs/sync/projects/autostim/analysis/iPScnn/config.yaml', 'r') as f:
                d = yaml.load(f.read())
        elif getpass.getuser() == "dpedr":
            with open('D:/iPScnn/config.yaml', 'r') as f:
                d = yaml.load(f.read())
        else:
            with open('/media/storage/iPScnn/config.yaml', 'r') as f:
                d = yaml.load(f.read())

        self.emgdir = d[0]['dataworks']['folders'][getpass.getuser()]['emgrawdir']

    def extract_details(self, subj, act='save'):
        """this function creates a list of details from all subjects, which may be used later for analyses; two
        options are possible as action ('act'): a) save (default) or b) 'return' """

        if (act != 'save') and (act != 'return'):
            raise ValueError(act + ' is not a valid definition of an action, please use "save" or "return" instead')

        details_temp = list()
        print("Extracting details ...", flush=True)
        for idx, pseud in enumerate(subj):
            details_temp.append([pseud, idx,
                                 self.datobj.frame_name['age'][self.datobj.idx_list[0][idx]],
                                 self.datobj.frame_name['gender'][self.datobj.idx_list[0][idx]],
                                 self.datobj.frame_name['updrs_off'][self.datobj.idx_list[0][idx]],
                                 self.datobj.frame_name['updrs_on'][self.datobj.idx_list[0][idx]],
                                 self.datobj.frame_name['updrs_diff'][self.datobj.idx_list[0][idx]],
                                 self.datobj.frame_name['ledd'][self.datobj.idx_list[0][idx]],
                                 self.datobj.frame_name['type'][self.datobj.idx_list[0][idx]]])

            details = pds.DataFrame(data=details_temp,
                                    columns=['Name', 'index', 'age', 'gender',
                                             'updrsOFF', 'updrsON', 'updrsDiff', 'ledd', 'type'])
        if act == 'save':
            details.to_csv(os.path.join(self.datobj.wdir + "/data/EMG/detailsEMG.csv"), mode='w', header=True)
        elif act == 'return':
            return details

    def saveEMGfeatall(self, emgfiles, cond):
        for idx, pseud in enumerate(emgfiles):
            filename = self.file_trunk + "EMGfeat" + self.task + "_" + pseud + "_" + cond + ".pkl"

            if not os.path.isfile(filename):
                datemg = list()

                for datname in emgfiles[pseud]:
                    dattemp = EMGfileworks.load_file(os.path.join(self.emgdir, pseud, datname))
                    datemg.append(dattemp)

                emg = pds.DataFrame(np.vstack(datemg), columns=["EMG_1", "EMG_2", "EMG_3", "EMG_4",
                                                                "EMG_5", "EMG_6", "EMG_7", "EMG_8"])

                if self.debug:
                    plt.figure()
                    plt.plot(emg["EMG_8"])
                    plt.ylabel('EMG channel \nactivity [in a.u.]')
                    plt.ylabel('Time [in sec.]')
                    plt.show()

                self.filt.apply_filter(emg)

                if self.debug:
                    plt.plot(emg["EMG_8"])
                    plt.show()

                file_features = self.datobj.wdir + "/preprocess/all_features.xml"
                df = preprocess.EMGfeatures.features_from_xml_on_df(file_features, emg)
                df.to_pickle(filename)

            else:
                print("EMG features for subj:" + pseud + " in the " + cond +
                      " condition already extracted, continuing with next subject")


class EMGpredict:

    def __init__(self):
        self.debug = False

    def normalized_confusion_matrix(self, cm):
        return cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues, ax=None):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        # cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        # classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = self.normalized_confusion_matrix(cm)
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        if ax is None:
            fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        # fig.tight_layout()
        return ax

    def prepare_pipeline(self, train_in: pds.DataFrame, train_out: pds.DataFrame,
                         predictor: str, norm_per_feature: bool = False,
                         **predictor_args):

        def rms_loss_func(y_true, y_pred):
            rms = np.sqrt(np.mean((y_true-y_pred)**2))
            return rms
        #rms_score = make_scorer(rms_loss_func, greater_is_better=False)
        rms_score = make_scorer(rms_loss_func, greater_is_better=True)

        if norm_per_feature:
            scaler = StandardScalerPerFeature()
        else:
            scaler = StandardScaler()

        if predictor == "LDA":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            predictor_instance = LinearDiscriminantAnalysis(**predictor_args)
            pipe = Pipeline([('scaler', scaler), ('predictor', predictor_instance)])
            # gparams not defined, function will exit with error
        elif predictor == "QDA":
            from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
            predictor_instance = QuadraticDiscriminantAnalysis(**predictor_args)
            pipe = Pipeline([('scaler', scaler), ('predictor', predictor_instance)])
            # gparams not defined, function will exit with error
        elif predictor == "kNN":
            from sklearn.neighbors import KNeighborsClassifier
            predictor_instance = KNeighborsClassifier(**predictor_args)
            pipe = Pipeline([('scaler', scaler), ('predictor', predictor_instance)])
            # gparams not defined, function will exit with error
        elif predictor == "SVM":
            from sklearn.svm import SVC
            predictor_instance = SVC(**predictor_args)
            pipe = Pipeline([('scaler', scaler), ('predictor', predictor_instance)])
            gparams = {'predictor__C': np.arange(1, 100, 5)}
        elif predictor == "SVR":
            from sklearn.svm import SVR
            predictor_instance = SVR(**predictor_args)
            pipe = Pipeline([('scaler', scaler), ('predictor', predictor_instance)])
            gparams = {'predictor__C': np.arange(1, 100, 5)}
        elif predictor == "LinearRegression":
            from sklearn.linear_model import LinearRegression
            predictor_instance = LinearRegression(**predictor_args)
            gparams = {'predictor__fit_intercept': [True, False]}
        elif predictor == "Lasso":
            from sklearn.linear_model import Lasso
            predictor_instance = Lasso(**predictor_args)
            gparams = {'predictor__alpha': np.arange(1, 10, .5)}
        elif predictor == "kNNRegression":
            from sklearn.neighbors import KNeighborsRegressor
            predictor_instance = KNeighborsRegressor(**predictor_args)
            gparams = {'predictor__n_neighbors': np.arange(1, 50)}
        else:
            raise valueerror(predictor + ' is not a valid predictor')

        # TODO: implement root mean square as optimization function via 'scoring' parameter
        # see e.g.: https://www.programcreek.com/python/example/89268/sklearn.metrics.make_scorer
        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring

        pipe = Pipeline([('scaler', scaler), ('predictor', predictor_instance)])
        gridsearch = GridSearchCV(pipe, gparams, scoring = rms_score)
        gridsearch.fit(train_in, train_out)
        # pipe.fit(train_in, train_out)
        return gridsearch


class StandardScalerPerFeature(StandardScaler):

    def fit(self, X: pds.DataFrame, y=None):
        features = [re.match(r"input_[0-9]+_([A-Z]+)_[0-9]+", l).group(1) for l in list(X)]
        unique_features = list(set(features))

        fit_data = pds.DataFrame(columns=features)

        for uf in unique_features:
            uf_data = X.filter(regex="input_[0-9]+_" + uf + "_[0-9]+").values.reshape(-1, 1).astype(float)
            fit_data[uf] = uf_data[:, 0]

        return super().fit(fit_data, y)