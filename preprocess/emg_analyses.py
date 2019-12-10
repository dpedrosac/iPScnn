#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import preprocess.dataworks
import preprocess.emg_features
import numpy as np
import pandas as pds
import yaml
import getpass
import matplotlib.pyplot as plt

datobj = preprocess.dataworks.DataPipeline('', 'tap', 'EMG', '')
preproc = preprocess.emg_features.EMGPreprocess()
timedomain = preprocess.emg_features.EMGFeaturesTime()
freqdomain = preprocess.emg_features.EMGFeaturesFreq()

t = datobj.tasks

dat_on = np.empty((len(datobj.subjlist), 31, 8))
dat_off = np.empty((len(datobj.subjlist), 31, 8))
for c in datobj.conds:
    details_temp = list()
    for idx, pseud in enumerate(datobj.subjlist):

        details_temp.append([pseud, c, t, idx,
                             datobj.frame_name['age'][datobj.idx_list[0][idx]],
                             datobj.frame_name['gender'][datobj.idx_list[0][idx]],
                             datobj.frame_name['updrs_off'][datobj.idx_list[0][idx]],
                             datobj.frame_name['updrs_on'][datobj.idx_list[0][idx]],
                             datobj.frame_name['updrs_diff'][datobj.idx_list[0][idx]],
                             datobj.frame_name['ledd'][datobj.idx_list[0][idx]]]
                            )

        list_files_emg = sorted(preproc.file_browser(str(pseud + "_" + t + "_" + c + "_EMG")))
        loaded_temp = list()
        datemg = list()

        for fix in range(len(list_files_emg)):
            dattemp = datobj.load_file(os.path.join(preproc.emgdir, list_files_emg[fix]))
            datemg.append(dattemp)

            if datobj.debug == True:
                plt.plot(datemg[:, 0])
                plt.ylabel('some numbers')
                plt.show()

        print("Processing subj:" + pseud + " in the " + c + " condition")
        rawemg = np.transpose(np.vstack(datemg))
        filteredEMG = preproc.butter_lowpass_filter(rawemg, 50, 200, 2)
        filteredEMG = preproc.butter_highpass_filter(filteredEMG, 20, 200, 2)

        rawEMGPowerSpectrum, frequencies = preproc.getPSD(filteredEMG)

        if datobj.debug == True:
            plt.plot(frequencies, rawEMGPowerSpectrum[3, :])
            plt.ylabel('PSD [in a.u]')
            plt.show()

        feat_temp = np.empty((31, filteredEMG.shape[0]))
        feat_temp[:] = np.nan
        for k in range(0, filteredEMG.shape[0], 1):
            feat_temp[0, k] = timedomain.getIEMG(filteredEMG[k, :])
            feat_temp[1, k] = timedomain.getMAV(filteredEMG[k,:])
            feat_temp[2, k] = timedomain.getMAV1(filteredEMG[k,:])
            feat_temp[3, k] = timedomain.getMAV2(filteredEMG[k, :])
            feat_temp[4, k] = timedomain.getSSI(filteredEMG[k, :])
            feat_temp[5, k] = timedomain.getVAR(filteredEMG[k, :])
            feat_temp[6, k] = timedomain.getTM(filteredEMG[k, :], 3)
            feat_temp[7, k] = timedomain.getTM(filteredEMG[k, :], 4)
            feat_temp[8, k] = timedomain.getTM(filteredEMG[k, :], 5)
            feat_temp[9, k] = timedomain.getLOG(filteredEMG[k, :])
            feat_temp[10, k] = timedomain.getRMS(filteredEMG[k, :])
            feat_temp[11, k] = timedomain.getWL(filteredEMG[k, :])
            feat_temp[12, k] = timedomain.getAAC(filteredEMG[k, :])
            feat_temp[13, k] = timedomain.getDASDV(filteredEMG[k, :])
            feat_temp[14, k] = timedomain.getAFB(filteredEMG[k, :], samplerate=preproc.samplerate, windowSize=32)
            feat_temp[15, k] = timedomain.getZC(filteredEMG[k, :], threshold=0.01)
            feat_temp[16, k] = timedomain.getMYOP(filteredEMG[k, :], threshold=0.01)
            feat_temp[17, k] = timedomain.getWAMP(filteredEMG[k, :], threshold=0.01)
            feat_temp[18, k] = timedomain.getSSC(filteredEMG[k, :], threshold=0.01)
            # feat_temp[19, k] = timedomain.getMAVSLPk(filteredEMG[k,:], nseg=50)
            # feat_temp[19, k] = timedomain.getHIST(filteredEMG[k,:], nseg=100, threshold=50)

            feat_temp[20, k] = freqdomain.getMNF(rawEMGPowerSpectrum[k, :], frequencies)
            feat_temp[21, k] = freqdomain.getMDF(rawEMGPowerSpectrum[k, :], frequencies)
            feat_temp[22, k] = freqdomain.getPeakFrequency(rawEMGPowerSpectrum[k, :], frequencies)
            feat_temp[23, k] = freqdomain.getMNP(rawEMGPowerSpectrum[k, :])
            feat_temp[24, k] = freqdomain.getTTP(rawEMGPowerSpectrum[k, :])
            feat_temp[25, k] = freqdomain.getSM(rawEMGPowerSpectrum[k, :], frequencies, order=1)
            feat_temp[26, k] = freqdomain.getSM(rawEMGPowerSpectrum[k, :], frequencies, order=2)
            feat_temp[27, k] = freqdomain.getSM(rawEMGPowerSpectrum[k, :], frequencies, order=3)
            feat_temp[28, k] = freqdomain.getFR(rawEMGPowerSpectrum[k, :], frequencies)
            feat_temp[29, k] = freqdomain.getPSR(rawEMGPowerSpectrum[k, :], frequencies)
            feat_temp[30, k] = freqdomain.getVCF(feat_temp[24, k], feat_temp[25, k], feat_temp[26, k])

        if c == 'ON':
            dat_on[idx, :, :] = np.reshape(feat_temp, [1, feat_temp.shape[0], feat_temp.shape[1]])
        else:
            dat_off[idx, :, :] = np.reshape(feat_temp, [1, feat_temp.shape[0], feat_temp.shape[1]])

list_ch = ["EMG%d" %i for i in range(1,9)]
for idx, n in enumerate(list_ch):
    np.savetxt(os.path.join(datobj.wdir + "/data/EMG/ONdataFeat_" + list_ch[idx] + ".csv"), dat_on[:, :, idx],
                           delimiter=";",
               header= "IEMG,MAV,MAV1,MAV2,SSI,VAR,TM3,TM4,TM5,LOG,RMS,WL,AAC,DASDV,AFB,ZC,MYOP,WAMP,SSC,MNF,MDF,PeakFreq,MNP,TTP,SM1,SM2,SM3,FR,PSR,VCF")
    np.savetxt(os.path.join(datobj.wdir + "/data/EMG/OFFdataFeat_" + list_ch[idx] + ".csv"), dat_off[:, :, idx],
                           delimiter=";",
               header= "IEMG,MAV,MAV1,MAV2,SSI,VAR,TM3,TM4,TM5,LOG,RMS,WL,AAC,DASDV,AFB,ZC,MYOP,WAMP,SSC,MNF,MDF,PeakFreq,MNP,TTP,SM1,SM2,SM3,FR,PSR,VCF")

details = pds.DataFrame(data=details_temp,
                          columns=['Name', 'condition', 'task', 'trial', 'age', 'gender',
                                   'updrsON', 'updrsOFF', 'updrsDiff', 'ledd'])
details.to_csv(os.path.join(datobj.wdir + "/data/EMG/detailsEMG.csv"), mode='w', header=True)
