#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import preprocess.dataworks
import preprocess.plotroutine
import numpy as np
import pandas as pds
import matplotlib.pyplot as plt
import pickle
import re
from scipy import stats
import math
import seaborn as sns
import operator


## Start estimation of data via <EMGanalyses>
task = 'tap'
device = 'EMG'

datobj = preprocess.dataworks.DataPipeline('', 'tap', 'EMG', '')
conds = ['OFF', 'ON']
file_trunk = datobj.wdir + "/data/EMG/features_subj/"
details = pds.read_csv(datobj.wdir + "/data/EMG/detailsEMG.csv")

MOI = ['IAV', 'MAV1', 'AAC', 'DASDV', 'SSC', 'WAMP', 'RMS', 'ZC', 'WL']
COI = [1,4,8] # channel of interest
idx1 = datobj.subjlist[1:13]
idx2 = datobj.subjlist[15:38]
datobj.subjlist = idx1.append(idx2)
idx1 = details[1:13]
idx2 = details[15:38]
details = idx1.append(idx2)

feat_on = pds.DataFrame(np.nan, index=range(1,datobj.subjlist.shape[0]+1), columns=MOI)
feat_off = pds.DataFrame(np.nan, index=range(1,datobj.subjlist.shape[0]+1), columns=MOI)

for c in conds:
    for idx_subj, pseud in enumerate(datobj.subjlist):
        print("Processing subj:" + pseud + " in the " + c + " condition")
        pickle_file = open(file_trunk + "EMGfeat_tap_" + pseud + "_" + c + ".pkl", 'rb')
        dftemp = pickle.load(pickle_file)

        for mtr in MOI:
            print("Extracting " + mtr + "...")
            dattemp = []
            cols = []

            regex = re.compile(mtr + "_" + str(list(COI)), re.IGNORECASE)
            cols = list(filter(regex.search, list(dftemp.columns.values)))
            dattemp = dftemp[cols]
            if c == 'ON':
                feat_on[mtr].iloc[idx_subj] = np.mean(dattemp.values.flatten())
            else:
                feat_off[mtr].iloc[idx_subj] = np.mean(dattemp.values.flatten())

# Start plotting routine group comparisons between ON and OFF
preprocess.plotroutine.PlotRoutines(MOI, feat_off, feat_on, ['OFF', 'ON'], ["OFF", "ON"], details, "bp", "dep")
preprocess.plotroutine.PlotRoutines(MOI, feat_off, feat_on, ['OFF', 'ON'], ["OFF", "ON"], details, "regr", "dep")

# Start with subjtype comparisons

groups = ['BR', 'TD', 'MX']
MOI = ['IAV', 'MAV1', 'AAC', 'DASDV', 'SSC', 'WAMP', 'RMS', 'ZC', 'WL']
COI = [1,2,4,5] # channel of interest

feat_br = pds.DataFrame(np.nan, index=range(np.size(np.where(details["type"]==0))), columns=MOI)
feat_td = pds.DataFrame(np.nan, index=range(np.size(np.where(details["type"]==1))), columns=MOI)
feat_mt = pds.DataFrame(np.nan, index=range(np.size(np.where(details["type"]==2))), columns=MOI)

for idx_g, g in enumerate(groups):
    idx_subj = np.where(details["type"]==idx_g)
    for pseud_no in range(np.size(idx_subj)):
        print("Processing subj:" + datobj.subjlist.iloc[idx_subj[0][pseud_no]] + " in the OFF condition")
        pickle_file = open(file_trunk + "EMGfeat_rst_" + datobj.subjlist.iloc[idx_subj[0][pseud_no]] + "_OFF.pkl", 'rb')
        dftemp = pickle.load(pickle_file)

        for mtr in MOI:
            print("Extracting " + mtr + "...")
            dattemp = []
            cols = []

            regex = re.compile(mtr + "_" + str(list(COI)), re.IGNORECASE)
            cols = list(filter(regex.search, list(dftemp.columns.values)))
            dattemp = dftemp[cols]
            if idx_g == 0:
                feat_br[mtr].iloc[pseud_no] = np.mean(dattemp.values.flatten())
            elif idx_g == 1:
                feat_td[mtr].iloc[pseud_no] = np.mean(dattemp.values.flatten())
            else:
                feat_mt[mtr].iloc[pseud_no] = np.mean(dattemp.values.flatten())


preprocess.plotroutine.PlotRoutines(MOI, feat_br, feat_td, ['BR', 'TD'],
                                    ["bradykinetic \nrigid", "tremor-\ndominant"], details, "bp", "indep")
