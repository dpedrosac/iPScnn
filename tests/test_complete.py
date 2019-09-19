import glob
import random
import pandas as pds
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical

filepath = '/media/david/windows/Dokumente und Einstellungen/dpedr/Jottacloud/onoff_svm'
conds = ['ON', 'OFF']
tasks = ['rest', 'hld', 'dd', 'tap']
dev = ['ACC', 'GYRO']


def import_xls(filepath):
    """imports the pseudonyms of the subjects to be processed in order to process the data accordingly"""
    os.chdir(filepath)
    frame_name = pds.DataFrame()
    filename = os.path.join(filepath + str("/patientenliste_onoff.xlsx"))
    frame_name = pds.read_excel(filename,
                                sheet_name='working')

    return frame_name.pseud[frame_name.group == 1]


def load_file(filename):
    """function that reads data and returns the values into a dataframe"""

    dataframe = pds.read_table(filename, header=None, sep='\s+')
    return preprocessing.normalize(preprocessing.scale(dataframe.values))

def standardize_data(dat):
    """function intended to standardise data, i.e. convert it to mean = 0 and std = 1; data should be as
    channel x time format, so that it can be looped per channel"""

    stand_dat = preprocessing.normalize(preprocessing.scale(dat))
    return stand_dat

def file_browser(datpath, word):
    """function which helps to find files with certain content, specified in the input"""
    file = []
    os.chdir(datpath)

    for f in glob.glob("*"):
        if word in f:
            file.append(f)

    return file


def load_all(frame_name):
    """loads all available data for the subjects in the list which was imported via (importXLS); At the end, there
    should be two files with a trial x time x sensor arrangement. The latter may include only ACC (size = 3),
    ACC+GYRO (size = 6) or ACC+GYRO+EMG (size=14), or any other combination"""

    datpath = filepath + "/analyses/csvdata/nopca/"
    loaded_on = list()
    loaded_off = list()
    for c in conds:
        loaded_temp = list()
        for t in tasks:
            for name in frame_name:
                list_files1 = file_browser(datpath, str(name + "_" + t + "_" + c + "_" + dev[0]))
                list_files2 = file_browser(datpath, str(name + "_" + t + "_" + c + "_" + dev[1]))

                for f, g in zip(sorted(list_files1), sorted(list_files2)):
                    datimu = list()
                    dattemp1 = load_file(os.path.join(datpath, f))
                    datimu.append(dattemp1)

                    dattemp2 = load_file(os.path.join(datpath, g))
                    datimu.append(dattemp2)

                    loaded_temp.append(np.hstack(datimu))
                    del datimu, dattemp1, dattemp2

        if c == 'ON':
            loaded_on = np.stack(loaded_temp, axis=0)
        else:
            loaded_off = np.stack(loaded_temp, axis=0)

    return loaded_on, loaded_off


def subsample_data(datON, datOFF, ratio):
    """in order to create a test and a validation dataset, this part just randomly samples a specific ratio of
    recordings and assigns them as test data; the output are two different datasets with all available data """

    trainingON_idx = np.random.randint(datON.shape[0], size=int(round(datON.shape[0] * ratio)))
    testON_idx = np.random.randint(datON.shape[0], size=int(round(datON.shape[0] * (1 - ratio))))
    smplsONtrain, smplsONtest = datON[trainingON_idx, :, :], datON[testON_idx, :, :]

    trainingOFF_idx = np.random.randint(datOFF.shape[0], size=int(round(datOFF.shape[0] * ratio)))
    testOFF_idx = np.random.randint(datOFF.shape[0], size=int(round(datOFF.shape[0] * (1 - ratio))))
    smplsOFFtrain, smplsOFFtest = datOFF[trainingOFF_idx, :, :], datOFF[testOFF_idx, :, :]

    return smplsONtrain, smplsONtest, smplsOFFtrain, smplsOFFtest


def create_cat(datON, datOFF):
    """this function establishes the categories for the data, i.e. whether it is an 'ON' or 'OFF' condition and
    concatenates all available recordings into a single matrix"""
    # TODO: at this point here, a selection of train and test data, possibly with a permutation could be included
    cats = np.zeros(datON.shape[0] + datOFF.shape[0])
    cats[0:datON.shape[0]] = 1
    cats = to_categorical(cats)
    datAll = np.concatenate([datON, datOFF])

    return datAll, cats


def evaluate_model_filt(trainX, trainy, testX, testy, n_filters):
    verbose, epochs, batch_size = 0, 10, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=n_filters, kernel_size=5, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=n_filters, kernel_size=5, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy


# fit and evaluate a model
def evaluate_model_kern(trainX, trainy, testX, testy, n_kernel):
    verbose, epochs, batch_size = 0, 15, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=n_kernel, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=64, kernel_size=n_kernel, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy


# summarize scores
def summarize_results_filt(scores, params):
    print(scores, params)

    for i in range(len(scores)):
        m, s = np.mean(scores[i]), np.std(scores[i])
        print('Param=%d: %.3f%% (+/-%.3f)' % (params[i], m, s))

    plt.boxplot(scores, labels=params)
    plt.savefig('exp_cnn_filters.png')


def run_experiment_filt(params, repeats=10):
    frame_name = import_xls(filepath)
    datON, datOFF = load_all(frame_name)
    smplsONtrain, smplsONtest, smplsOFFtrain, smplsOFFtest = subsample_data(datON, datOFF, .25)
    trainX, trainy = create_cat(smplsONtest, smplsOFFtest)
    testX, testy = create_cat(smplsONtrain, smplsOFFtrain)

    all_scores = list()
    for p in params:
        scores = list()
        for r in range(repeats):
            score = evaluate_model_filt(trainX, trainy, testX, testy, p)
            score = score * 100.0
            print('>p=%d #%d: %.3f' % (p, r+1, score))
            scores.append(score)
        all_scores.append(scores)
    summarize_results_filt(all_scores, params)


# summarize scores
def summarize_results_kern(scores, params):
    print(scores, params)
    # summarize mean and standard deviation
    for i in range(len(scores)):
        m, s = np.mean(scores[i]), np.std(scores[i])
        print('Param=%d: %.3f%% (+/-%.3f)' % (params[i], m, s))
    # boxplot of scores
    plt.boxplot(scores, labels=params)
    plt.savefig('exp_cnn_kernel.png')


# run an experiment
def run_experiment_kern(params, repeats=10):
    frame_name = import_xls(filepath)
    datON, datOFF = load_all(frame_name)
    smplsONtrain, smplsONtest, smplsOFFtrain, smplsOFFtest = subsample_data(datON, datOFF, .25)
    trainX, trainy = create_cat(smplsONtest, smplsOFFtest)
    testX, testy = create_cat(smplsONtrain, smplsOFFtrain)

    all_scores = list()
    for p in params:
        # repeat experiment
        scores = list()
        for r in range(repeats):
            score = evaluate_model_kern(trainX, trainy, testX, testy, p)
            score = score * 100.0
            print('>p=%d #%d: %.3f' % (p, r + 1, score))
            scores.append(score)
        all_scores.append(scores)
    # summarize results
    summarize_results_kern(all_scores, params)

# run the experiment
#n_params = [32, 64, 256]
#run_experiment_filt(n_params)
n_kern = [2, 3, 5, 7, 11]
run_experiment_kern(n_kern)

## Debug results
def plot_rawdata(dat):
    time_vector = np.linspace(0, dat.shape[1] / 50, num=dat.shape[1])
    samples = random.sample(list(range(1, dat.shape[0])), 15)
    plt.figure()
    for y in samples:
        plt.subplot(1,2,1)
        plt.plot(time_vector, dat[y, :, 0])
        plt.subplot(1,2,2)
        plt.plot(time_vector, dat[y, :, 3])

    plt.show()

def plot_variable_distributions(dat):
    # flatten windows
    longX = dat.reshape((dat.shape[0] * dat.shape[1], dat.shape[2]))
    print(longX.shape)
    plt.figure()
    xaxis = None
    for i in range(longX.shape[1]):
        ax = plt.subplot(longX.shape[1], 1, i + 1, sharex=xaxis)
        ax.set_xlim(-1, 1)
        if i == 0:
            xaxis = ax
        plt.hist(longX[:, i], bins=100)
    plt.show()
