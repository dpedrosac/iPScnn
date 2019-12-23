#!/usr/bin/env python
# -*- coding: utf-8 -*-

import preprocess.EMGpreprocess
import preprocess.EMGfeatures
import preprocess.EMGfilter

import os
import preprocess.dataworks
import preprocess.emg_features
import numpy as np
import pandas as pds
import matplotlib.pyplot as plt
import pickle

class ExtractData:
    """functions required to etxract EMG data in order to process it later"""

    def __init__(self, task, filename_append):
        self.debug = False
        self.device = 'EMG'
        self.task = task
        self.datobj = preprocess.dataworks.DataPipeline('', self.task, self.device, '')
        self.preproc = preprocess.emg_features.EMGPreprocess()
        self.filt = preprocess.EMGfilter.Filter()
        if filename_append == '':
            self.file_trunk = self.datobj.wdir + "/data/EMG/features_subj/"
        else:
            self.file_trunk = self.datobj.wdir + "/data/EMG/" + filename_append

    def extractEMGfeat(self):

        for c in self.datobj.conds:
            details_temp = list()
            for idx, pseud in enumerate(self.datobj.subjlist):
                print("Processing subj:" + pseud + " in the " + c + " condition")
                filename = self.file_trunk + "EMGfeat" + self.task + "_" + pseud + "_" + c + ".pkl"

                # TODO: details only when all; add option for train/test dataset; add option for k-fold validation
                details_temp.append([pseud, c, self.task, idx,
                                     self.datobj.frame_name['age'][self.datobj.idx_list[0][idx]],
                                     self.datobj.frame_name['gender'][self.datobj.idx_list[0][idx]],
                                     self.datobj.frame_name['updrs_off'][self.datobj.idx_list[0][idx]],
                                     self.datobj.frame_name['updrs_on'][self.datobj.idx_list[0][idx]],
                                     self.datobj.frame_name['updrs_diff'][self.datobj.idx_list[0][idx]],
                                     self.datobj.frame_name['ledd'][self.datobj.idx_list[0][idx]],
                                     self.datobj.frame_name['type'][self.datobj.idx_list[0][idx]]])

                if not os.path.isfile(filename):
                    list_files_emg = sorted(self.preproc.file_browser(str(pseud + "_" + self.task + "_" + c + "_EMG")))
                    loaded_temp = list()
                    datemg = list()

                    for fix in range(len(list_files_emg)):
                        dattemp = datobj.load_file(os.path.join(preproc.emgdir, list_files_emg[fix]))
                        datemg.append(dattemp)

                        if datobj.debug:
                            plt.plot(datemg[:, 0])
                            plt.ylabel('some numbers')
                            plt.show()

                    emg = pds.DataFrame(np.vstack(datemg), columns=["EMG_1", "EMG_2", "EMG_3", "EMG_4",
                                                                    "EMG_5","EMG_6","EMG_7","EMG_8"])

                    if datobj.debug:
                        plt.figure()
                        plt.plot(emg["EMG_1"])
                        plt.ylabel('some numbers')
                        plt.show()

                    filt.apply_filter(emg)

                    if datobj.debug:
                        plt.figure()
                        plt.plot(emg["EMG_1"])
                        plt.ylabel('some numbers')
                        plt.show()

                    file_features = datobj.wdir + "/preprocess/all_features.xml"
                    df = preprocess.EMGfeatures.features_from_xml_on_df(file_features, emg)
                    df.to_pickle(filename)
                else:
                    print("EMG features for subj:" + pseud + " in the " + c + " condition already "
                                                                              "extracted, continuing with next subject")

        details = pds.DataFrame(data=details_temp,
                                  columns=['Name', 'condition', 'task', 'trial', 'age', 'gender',
                                           'updrsON', 'updrsOFF', 'updrsDiff', 'ledd', 'type'])
        details.to_csv(os.path.join(datobj.wdir + "/data/EMG/detailsEMG.csv"), mode='w', header=True)