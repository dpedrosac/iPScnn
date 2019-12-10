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


datobj = preprocess.dataworks.DataPipeline('', 'tap', 'EMG', '')
preproc = preprocess.emg_features.EMGPreprocess()
filt = preprocess.EMGfilter.Filter()

t = datobj.tasks
file_trunk = datobj.wdir + "/data/EMG/features_subj/"

for c in datobj.conds:
    details_temp = list()
    for idx, pseud in enumerate(datobj.subjlist):
        print("Processing subj:" + pseud + " in the " + c + " condition")
        filename = file_trunk + "EMGfeat_" + pseud + "_" + c + ".pkl"

        details_temp.append([pseud, c, t, idx,
                             datobj.frame_name['age'][datobj.idx_list[0][idx]],
                             datobj.frame_name['gender'][datobj.idx_list[0][idx]],
                             datobj.frame_name['updrs_off'][datobj.idx_list[0][idx]],
                             datobj.frame_name['updrs_on'][datobj.idx_list[0][idx]],
                             datobj.frame_name['updrs_diff'][datobj.idx_list[0][idx]],
                             datobj.frame_name['ledd'][datobj.idx_list[0][idx]]]
                            )

        if os.path.isfile(filename) == False:
            list_files_emg = sorted(preproc.file_browser(str(pseud + "_" + t + "_" + c + "_EMG")))
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
                                   'updrsON', 'updrsOFF', 'updrsDiff', 'ledd'])
details.to_csv(os.path.join(datobj.wdir + "/data/EMG/detailsEMG.csv"), mode='w', header=True)