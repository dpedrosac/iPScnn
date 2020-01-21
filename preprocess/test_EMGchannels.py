import pickle, glob, os
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


folder = "C:/Users/dpedr/Desktop/test_EMGchannels"
file = "stnL_c1_5.0mA_dd_20200109-165832.pkl"

fig = plt.figure()

for k in range(1,9):
    fig, axs = plt.subplots(8, sharex=True)
    for name in glob.glob(folder+'/*c'+str(k)+'*dd*'):
        with open(name, 'rb') as fp:
            imu_data = pickle.load(fp)
            emg_data = pickle.load(fp)

        # EMG data
        if len(imu_data[0]) == 8:
            emg_all = np.array(imu_data)
        else:
            emg_all = np.array(emg_data)

        time_emg = np.linspace(0, len(emg_all[:, 0]), len(emg_all[:, 0])) / 200

        for ch in range(0,8):
            axs[ch].plot(time_emg, emg_all[:,ch])
        plt.show()