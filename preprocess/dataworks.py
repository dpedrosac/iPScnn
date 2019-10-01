#!/usr/bin/env python
# -*- coding: utf-8 -*-


import glob
import pandas as pds
import numpy as np
import os
import matplotlib.pyplot as plt

from keras.utils import to_categorical


class DataPipeline:
    def __init__(self, fpath, conds, tasks, dev, ignbad):
        self._ignbad = ignbad
        self.scaling = True

        # define the standard path, conditions, tasks and devices whenever 'all is selected'
        if fpath == '':
            self.wdir = '/media/david/windows/Dokumente und Einstellungen/dpedr/Jottacloud/onoff_svm'
        else:
            self.wdir = fpath

        if conds == 'all':
            self.conds = ['ON', 'OFF']
        else:
            self.conds = conds

        if tasks == 'all':
            self.tasks = ['rest', 'hld', 'dd', 'tap']
        else:
            self.tasks = tasks

        if dev == 'all':
            self.dev = ['ACC', 'GYRO']
        else:
            self.dev = dev

    def generate_subjlist(self):
        """imports the pseudonyms of the subjects to be processed in order to later read the data accordingly"""
        os.chdir(self.wdir)
        filename = os.path.join(self.wdir + str("/patientenliste_onoff.xlsx"))
        # frame_name = pds.DataFrame()
        frame_name = pds.read_excel(filename,
                                    sheet_name='working')

        # ignores the data categorised as wrong/bad from the dataset (see patienten_onoff.xls for details)
        if self._ignbad:
            self.subjlist = frame_name.pseud[frame_name.group == 1]
        else:
            self.subjlist = frame_name.pseud

    def load_all(self):
        """loads all available data for the subjects in the list which was imported via (importXLS); At the end, there
        should be two files with a trial x time x sensor arrangement. The latter may include only ACC (size = 3),
        ACC+GYRO (size = 6) or ACC+GYRO+EMG (size=14), or any other combination"""

        self.datpath = self.wdir + "/analyses/csvdata/nopca/"
        loaded_on = list()
        loaded_off = list()

        # loop through conditions ('ON', 'OFF'), tasks ('rst', 'hld', 'dd', 'tap') and subjects to get all data
        for c in self.conds:
            loaded_temp = list()
            for t in self.tasks:
                for name in self.subjlist:
                    list_files1 = self.file_browser(str(name + "_" + t + "_" + c + "_" + self.dev[0]))
                    list_files2 = self.file_browser(str(name + "_" + t + "_" + c + "_" + self.dev[1]))

                    for f, g in zip(sorted(list_files1), sorted(list_files2)):
                        datimu = list()
                        dattemp1 = self.load_file(os.path.join(self.datpath, f))
                        datimu.append(dattemp1)

                        dattemp2 = self.load_file(os.path.join(self.datpath, g))
                        datimu.append(dattemp2)

                        loaded_temp.append(np.hstack(datimu))
                        del datimu, dattemp1, dattemp2

            if c == 'ON':
                loaded_on = np.stack(loaded_temp, axis=0)
            else:
                loaded_off = np.stack(loaded_temp, axis=0)

        return loaded_on, loaded_off

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


# TODO At this point, one should think about a series of different plotting possibilities aiming at providing
#  control/sanity checks for the data reading and processing

# class PlotData:
#    def __init__(self):
#        pass
