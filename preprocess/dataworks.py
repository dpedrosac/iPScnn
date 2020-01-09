#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import pandas as pds
import numpy as np
import os
import getpass
import scipy
from keras.utils import to_categorical
import yaml

class DataPipeline:
    def __init__(self, conds, tasks, dev):

        # load local settings.
        # TODO ugly because we use a hardcoded path here and call this exact code snippet also in iPScnn.py
        # This should be dealt with in an own library e.g. "utils.load_settings"
        if getpass.getuser() == "urs":
            with open('/home/urs/sync/projects/autostim/analysis/iPScnn/config.yaml', 'r') as f:
                d = yaml.load(f.read())
        else:
            with open('/media/storage/iPScnn/config.yaml', 'r') as f:
                d = yaml.load(f.read())

        self._ignbad = d[0]['dataworks']['ignbad']
        self.scaling = d[0]['dataworks']['ignbad']
        self.datpath = d[0]['dataworks']['folders'][getpass.getuser()]['datpath']
        self.patlist = d[0]['dataworks']['folders'][getpass.getuser()]['patlist']
        self.debug = False

        # define the standard path, conditions, tasks and devices whenever 'all is selected'
        self.wdir = d[0]['dataworks']['folders'][getpass.getuser()]['wdir']

        if conds == '':
            self.conds = d[0]['dataworks']['conds']
        elif conds == 'all':
            self.conds = ['ON', 'OFF']
        else:
            self.conds = conds

        if tasks == '':
            self.tasks = d[0]['dataworks']['tasks']
        elif tasks == 'all':
            self.tasks = ['rst', 'hld', 'dd', 'tap']
        else:
            self.tasks = tasks

        if dev == '':
            self.dev = d[0]['dataworks']['dev']
        elif dev == 'all':
            self.dev = ['ACC', 'GYRO']
        else:
            self.dev = dev

    def generate_subjlist(self):
        """imports the pseudonyms of the subjects to be processed in order to later read the data accordingly"""
        os.chdir(self.wdir)
        filename = self.patlist
        #if getpass.getuser() == 'urs':
        #    filename = os.path.join(self.wdir + str("/data/patientenliste_onoff.xlsx"))
        #else:
        #    filename = os.path.join(self.wdir + str("/patientenliste_onoff.xlsx"))

        self.frame_name = pds.read_excel(filename,
                                    sheet_name='working')

        # ignores the data categorised as wrong/bad from the dataset (see patienten_onoff.xls for details)
        if self._ignbad:
            self.subjlist = self.frame_name.pseud[self.frame_name.group == 1]
            self.idx_list = np.where(self.frame_name['group'] == 1)
        else:
            self.subjlist = self.frame_name.pseud

    def load_all(self, window=False):
        """loads all available data for the subjects in the list which was imported via (importXLS); At the end, there
        should be two files with a trial x time x sensor arrangement. The latter may include only ACC (size = 3),
        ACC+GYRO (size = 6) or ACC+GYRO+EMG (size=14), or any other combination"""
        #if getpass.getuser() == 'urs':
        #    self.datpath = self.wdir + "/data/csvdata/nopca/"
        #else:
        #    self.datpath = self.wdir + "/analyses/csvdata/nopca/"
        loaded_on = list()
        loaded_off = list()

        # loop through conditions, tasks and subjects to get all data; for details see (config.yaml)
        for c in self.conds:
            loaded_temp = list()
            details_temp = list()
            for t in self.tasks:
                for idx, name in enumerate(self.subjlist):
                    list_files_acc  = sorted(self.file_browser(str(name + "_" + t + "_" + c + "_ACC" )))
                    list_files_gyro = sorted(self.file_browser(str(name + "_" + t + "_" + c + "_GYRO")))
                    list_files_emg  = sorted(self.file_browser(str(name + "_" + t + "_" + c + "_EMG" )))

                    for fix in range(len(list_files_acc)):
                        datimu    = list()
                        datlength = None

                        if 'EMG' in self.dev:
                            dattemp = self.load_file(os.path.join(self.datpath, list_files_emg[fix]))
                            datimu.append(dattemp)
                            datlength = dattemp.shape[0]

                        if 'ACC' in self.dev:
                            print(os.path.join(self.datpath, list_files_acc[fix]))
                            dattemp = self.load_file(os.path.join(self.datpath, list_files_acc[fix]))

                            if datlength != None:
                                dattemp = self.arrange_data(dattemp, datlength)

                            datimu.append(dattemp)

                        if 'GYRO' in self.dev:
                            dattemp = self.load_file(os.path.join(self.datpath, list_files_gyro[fix]))

                            if datlength != None:
                                x = np.arange(0, dattemp.shape[0])
                                fit = scipy.interpolate.interp1d(x, dattemp, axis=0)
                                dattemp = fit(np.linspace(0, dattemp.shape[0] - 1, datlength))

                            datimu.append(dattemp)
                            details_temp.append([name, c, t, fix+1,
                                                 self.frame_name['age'][self.idx_list[0][idx]],
                                                 self.frame_name['gender'][self.idx_list[0][idx]],
                                                 self.frame_name['updrs_off'][self.idx_list[0][idx]],
                                                 self.frame_name['updrs_on'][self.idx_list[0][idx]],
                                                 self.frame_name['updrs_diff'][self.idx_list[0][idx]],
                                                 self.frame_name['ledd'][self.idx_list[0][idx]]]
                                                )

                        loaded_temp.append(np.hstack(datimu))
                        #del datimu, dattemp, datlength, fit, x # doesnt't work, as apparently only needed when EMG data is processed?!?

            if c == 'ON':
                loaded_on = np.stack(loaded_temp, axis=0)
                detailsON = pds.DataFrame(data=details_temp,
                                          columns=['Name', 'condition', 'task', 'trial', 'age', 'gender',
                                                   'updrsON', 'updrsOFF', 'updrsDiff', 'ledd'])
            else:
                loaded_off = np.stack(loaded_temp, axis=0)
                detailsOFF = pds.DataFrame(data=details_temp,
                                          columns=['Name', 'condition', 'task', 'trial', 'age', 'gender',
                                                   'updrsON', 'updrsOFF', 'updrsDiff', 'ledd'])

        if window != False:
            # chop available data into time window pieces thus creating more samples
            # TODO:
            #   - implement sanity/type check for window values
            #   - implement overlap option for window

            windowed_on  = loaded_on [:, 0:window, :]
            windowed_off = loaded_off[:, 0:window, :]

            for n in range(window * 2, int(np.floor(loaded_on.shape[1] / window) * window) + 1, window):
                windowed_on  = np.append(windowed_on , loaded_on [:, (n - window):n, :], 0)
                windowed_off = np.append(windowed_off, loaded_off[:, (n - window):n, :], 0)

            loaded_on  = windowed_on
            loaded_off = windowed_off

        return loaded_on, loaded_off, detailsON, detailsOFF

    def arrange_data(self, dattemp, datlength):
        """helper function that interpolates data to make "EMG" and "IMU" data of same length """

        x = np.arange(0, dattemp.shape[0])
        fit = scipy.interpolate.interp1d(x, dattemp, axis=0)
        dattemp = fit(np.linspace(0, dattemp.shape[0] - 1, datlength))
        return dattemp

    def sliding_window(seq,winsize,step=1):
        """ Returns generator iterating through entire input."""

        # Verify the inputs
        try: it = iter(seq)
        except TypeError:
            raise Exception("Please make sure input is iterable.")
        if not ((type(winsize) == type(0)) and (type(step) == type(0))):
            raise Exception("Size of windows (winsize) and step must be integers.")
        if step > winsize:
            raise Exception("Steps may never be larger than winSize.")
        if winsize > len(seq):
            raise Exception("winsize must not be larger than sequence length.")

        # Pre-compute number of chunks to emit
        numchunks = ((len(seq)-winsize)/step)+1

        # Generare chunks of data
        for i in range(0,numchunks*step,step):
            yield seq[i:i+winsize]

    def load_file(self, filename):
        """ helper function that reads data and returns the values into a dataframe; there are two options:
        1)  True: Mean = 0 and Std = 1
        2)  False: Mean = 0, Std remains """
        from sklearn import preprocessing

        # read data from txt-file as processed via MATLAB
        dataframe = pds.read_table(filename, header=None, sep='\s+')

        # plot only if debug option is set.
        if self.debug:
            import matplotlib.pyplot as plt
            scaler = preprocessing.StandardScaler()
            plt.figure()
            plt.subplot(121)
            plt.plot(dataframe)

        if self.scaling:
            if self.debug:
                plt.subplot(122)
                plt.plot(preprocessing.robust_scale(dataframe.values))
                plt.show()
                plt.clf()
            return preprocessing.robust_scale(dataframe.values)
        else:
            return preprocessing.normalize(dataframe.values)

    def file_browser(self, word):
        """function which helps to find files with certain content, specified in the input"""
        file = []
        os.chdir(self.datpath)

        for f in glob.glob("*"):
            if word in f:
                file.append(f)

        return file

class Categorize:
    def __init__(self):
        pass

    def subsample_data(self, datON, datOFF, detailsON, detailsOFF, modeltype, ratio, tasks):
        """in order to create a test and a validation dataset, this part just randomly samples a specific ratio of
        recordings and assigns them as test data; the output are two different datasets with all available data.
        also for multihead models data is splitted into task specific matrices
        TODO there are two conditionals on modeltype necessary. merge?"""

        if modeltype == 'mh':
            # data preprocessing for multihead model
            datONout = list()
            detailsONout = list()
            datOFFout = list()
            detailsOFFout = list()
            cnt = 0
            outONlen = []
            outOFFlen = []
            for task in np.unique(detailsON.task):
                # split data into task specific matrices
                datONout.append(datON [detailsON ['task'] == task, :, :])
                datOFFout.append(datOFF[detailsOFF['task'] == task, :, :])
                detailsONout.append( detailsON [detailsON ['task'] == task])
                detailsOFFout.append(detailsOFF[detailsOFF['task'] == task])
                outONlen.append( datONout [cnt].shape[0])
                outOFFlen.append(datOFFout[cnt].shape[0])
                cnt = cnt + 1
            minOFFlen = min(outOFFlen)
            minONlen  = min(outONlen)
            cnt = 0
            for task in tasks:
                # shave all task matrices to same length
                # TODO: this will always delete data of the last subject.
                #       could be more sensible to randomize data first
                datONout [cnt] = datONout [cnt][0:minONlen , :, :]
                datOFFout[cnt] = datOFFout[cnt][0:minOFFlen, :, :]
                detailsONout [cnt] = detailsONout [cnt][0:minONlen]
                detailsOFFout[cnt] = detailsOFFout[cnt][0:minOFFlen]
                cnt = cnt +1
        else:
            minONlen  = datON.shape [0]
            minOFFlen = datOFF.shape[0]

        trainingON_idx  = np.random.randint(minONlen , size=int(round(minONlen  * ratio)))
        trainingOFF_idx = np.random.randint(minOFFlen, size=int(round(minOFFlen * ratio)))
        testON_idx  = np.setdiff1d(np.arange(minONlen) , trainingON_idx )
        testOFF_idx = np.setdiff1d(np.arange(minOFFlen), trainingOFF_idx)

        if modeltype == 'mh':
            cnt = 0
            smplsONtrain = list()
            smplsONtest = list()
            smplsOFFtrain = list()
            smplsOFFtest = list()
            for task in tasks:
                smplsONtrain.append( datONout [cnt][trainingON_idx  , :, :])
                smplsONtest.append(  datONout [cnt][testON_idx      , :, :])
                smplsOFFtrain.append(datOFFout[cnt][trainingOFF_idx , :, :])
                smplsOFFtest.append( datOFFout[cnt][testOFF_idx     , :, :])
                cnt = cnt + 1
        else:
            smplsONtrain,  smplsONtest  = datON [trainingON_idx , :, :], datON [testON_idx , :, :]
            smplsOFFtrain, smplsOFFtest = datOFF[trainingOFF_idx, :, :], datOFF[testOFF_idx, :, :]

        return smplsONtrain, smplsONtest, smplsOFFtrain, smplsOFFtest

    def create_cat(self, X, Y, modeltype):
        """this function establishes the categories for the data, i.e. whether it is an 'ON' or 'OFF' condition and
        concatenates all available recordings into a single matrix"""
        if modeltype == "mh":
            cats = np.zeros(X[0].shape[0] + Y[0].shape[0])
            cats[0:X[1].shape[0]] = 1
            cats = to_categorical(cats)
            datAll = list()
            for i in range(0,len(X)):
                datAll.append(np.concatenate([X[i], Y[i]]))
        else:
            cats = np.zeros(X.shape[0] + Y.shape[0])
            cats[0:X.shape[0]] = 1
            cats = to_categorical(cats)
            datAll = np.concatenate([X, Y])

        return datAll, cats