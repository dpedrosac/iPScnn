#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import preprocess.dataworks
import preprocess.emg_features
import numpy as np
import pandas as pds
import matplotlib.pyplot as plt
import pickle
import re
from scipy import stats
import math
import seaborn as sns

datobj = preprocess.dataworks.DataPipeline('', 'tap', 'EMG', '')
conds = ['OFF', 'ON']
file_trunk = datobj.wdir + "/data/EMG/features_subj/"
details = pds.read_csv(datobj.wdir + "/data/EMG/detailsEMG.csv")

MOI = ['IAV', 'MAV1', 'AAC', 'DASDV', 'SSC', 'WAMP', 'RMS', 'ZC', 'WL']
COI = [1,4] # channel of interest
feat_on = pds.DataFrame(np.nan, index=range(1,39), columns=MOI)
feat_off = pds.DataFrame(np.nan, index=range(1,39), columns=MOI)

for c in conds:
    for idx_subj, pseud in enumerate(datobj.subjlist):
        print("Processing subj:" + pseud + " in the " + c + " condition")
        pickle_file = open(file_trunk + "EMGfeat_" + pseud + "_" + c + ".pkl", 'rb')
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


# Start plotting routine and estimation of correlations

dims = math.ceil(np.sqrt(len(MOI)))
fig, ax = plt.subplots(dims, dims)
iter = -1

for i in range(int(dims)):
    for j in range(int(dims)):
        iter = iter + 1
        print(i, j, iter)

        try:
            mtr = MOI[iter]
            dataplot = pds.DataFrame([feat_on[mtr].values, feat_off[mtr].values], index=['ON', 'OFF']).transpose()

            sns.boxplot(data=dataplot,
                            width=0.5,
                            notch=False,
                            palette="Blues",
                            ax=ax[i, j]).set(ylabel=mtr)

            sns.swarmplot(data=dataplot,
                        color='k', # Make points black
                        alpha=0.7,
                        ax=ax[i, j]) # and slightly transparent


            # Adjust boxplot and whisker line properties
            #for p, artist in enumerate(ax.artists):
            #    artist.set_edgecolor('black')
            #    for q in range(p*6, p*6+6):
            #        line = ax.lines[q]
            #        line.set_color('black')

            [h, p] = stats.ttest_rel(feat_on[mtr].values, feat_off[mtr].values)
            print(p)

        except:
            print("Continuing")
