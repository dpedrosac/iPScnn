#!/usr/bin/env python
# -*- coding: utf-8 -*-


import glob
import pandas as pds
import numpy as np
import os
import getpass
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import scipy

from keras.utils import to_categorical


class DataPipeline:
    def __init__(self, fpath, conds, tasks, dev, ignbad):
        self._ignbad = ignbad
        self.scaling = True

        # define the standard path, conditions, tasks and devices whenever 'all is selected'
        if fpath == '':
            if getpass.getuser() == 'urs':
                self.wdir = '/home/urs/sync/projects/autostim'
            else:
                self.wdir = '/media/david/windows/Dokumente und Einstellungen/dpedr/Jottacloud/onoff_svm'
        else:
            self.wdir = fpath

        if conds == 'all':
            self.conds = ['ON', 'OFF']
        else:
            self.conds = conds

        if tasks == 'all':
            self.tasks = ['rst', 'hld', 'dd', 'tap']
        else:
            self.tasks = tasks

        if dev == 'all':
            self.dev = ['ACC', 'GYRO']
        else:
            self.dev = dev

    def generate_subjlist(self):
        """imports the pseudonyms of the subjects to be processed in order to later read the data accordingly"""
        os.chdir(self.wdir)
        if getpass.getuser() == 'urs':
            filename = os.path.join(self.wdir + str("/data/patientenliste_onoff.xlsx"))
        else:
            filename = os.path.join(self.wdir + str("/patientenliste_onoff.xlsx"))
        # frame_name = pds.DataFrame()
        frame_name = pds.read_excel(filename,
                                    sheet_name='working')

        # ignores the data categorised as wrong/bad from the dataset (see patienten_onoff.xls for details)
        if self._ignbad:
            self.subjlist = frame_name.pseud[frame_name.group == 1]
        else:
            self.subjlist = frame_name.pseud

    def load_all(self, window = False):
        """loads all available data for the subjects in the list which was imported via (importXLS); At the end, there
        should be two files with a trial x time x sensor arrangement. The latter may include only ACC (size = 3),
        ACC+GYRO (size = 6) or ACC+GYRO+EMG (size=14), or any other combination"""
        if getpass.getuser() == 'urs':
            self.datpath = self.wdir + "/data/csvdata/nopca/"
        else:
            self.datpath = self.wdir + "/analyses/csvdata/nopca/"
        loaded_on = list()
        loaded_off = list()

        # loop through conditions ('ON', 'OFF'), tasks ('rst', 'hld', 'dd', 'tap') and subjects to get all data
        for c in self.conds:
            loaded_temp = list()
            for t in self.tasks:
                for name in self.subjlist:
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
                            dattemp = self.load_file(os.path.join(self.datpath, list_files_acc[fix]))

                            if datlength != None:
                                # interpolate data if emg is present as emg is longer.
                                # TODO should be implemented as a function because it's used again below
                                x = np.arange(0, dattemp.shape[0])
                                fit = scipy.interpolate.interp1d(x, dattemp, axis=0)
                                dattemp = fit(np.linspace(0, dattemp.shape[0] - 1, datlength))

                            datimu.append(dattemp)

                        if 'GYRO' in self.dev:
                            dattemp = self.load_file(os.path.join(self.datpath, list_files_gyro[fix]))

                            if datlength != None:
                                x = np.arange(0, dattemp.shape[0])
                                fit = scipy.interpolate.interp1d(x, dattemp, axis=0)
                                dattemp = fit(np.linspace(0, dattemp.shape[0] - 1, datlength))

                            datimu.append(dattemp)

                        loaded_temp.append(np.hstack(datimu))
                        # doesnt't work, as apparently only needed when EMG data is processed?!?
                        #del datimu, dattemp, datlength, fit, x

            if c == 'ON':
                loaded_on = np.stack(loaded_temp, axis=0)
            else:
                loaded_off = np.stack(loaded_temp, axis=0)

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

        return loaded_on, loaded_off

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

        if self.scaling is True:
            return preprocessing.scale(preprocessing.normalize(dataframe.values))
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

    def subsample_data(self, datON, datOFF, ratio):
        """in order to create a test and a validation dataset, this part just randomly samples a specific ratio of
        recordings and assigns them as test data; the output are two different datasets with all available data """

        trainingON_idx = np.random.randint(datON.shape[0], size=int(round(datON.shape[0] * ratio)))
        #testON_idx = np.random.randint(datON.shape[0], size=int(round(datON.shape[0] * (1 - ratio))))
        testON_idx = np.setdiff1d(np.arange(datON.shape[0]), trainingON_idx)
        smplsONtrain, smplsONtest = datON[trainingON_idx, :, :], datON[testON_idx, :, :]

        trainingOFF_idx = np.random.randint(datOFF.shape[0], size=int(round(datOFF.shape[0] * ratio)))
        #testOFF_idx = np.random.randint(datOFF.shape[0], size=int(round(datOFF.shape[0] * (1 - ratio))))
        testOFF_idx = np.setdiff1d(np.arange(datOFF.shape[0]), trainingOFF_idx)
        smplsOFFtrain, smplsOFFtest = datOFF[trainingOFF_idx, :, :], datOFF[testOFF_idx, :, :]

        return smplsONtrain, smplsONtest, smplsOFFtrain, smplsOFFtest

    def create_cat(self, X, Y):
        """this function establishes the categories for the data, i.e. whether it is an 'ON' or 'OFF' condition and
        concatenates all available recordings into a single matrix"""
        cats = np.zeros(X.shape[0] + Y.shape[0])
        cats[0:X.shape[0]] = 1
        cats = to_categorical(cats)
        datAll = np.concatenate([X, Y])

        return datAll, cats
