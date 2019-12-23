#!/usr/bin/env python
# -*- coding: utf-8 -*-

import peakutils, os, glob
import numpy as np
import getpass
import math
from scipy.signal import butter, lfilter, welch, square  # for signal filtering
import yaml


class EMGPreprocess:
    """Functions which help to preprocess the EMG data properly"""

    def __init__(self):
        if getpass.getuser() == "urs":
            with open('/home/urs/sync/projects/autostim/analysis/iPScnn/config.yaml', 'r') as f:
                d = yaml.load(f.read())
        elif getpass.getuser() == "dpedr":
            with open('D:/iPScnn/config.yaml', 'r') as f:
                d = yaml.load(f.read())
        else:
            with open('/media/storage/iPScnn/config.yaml', 'r') as f:
                d = yaml.load(f.read())

        # TODO insert a list of features to extract in the yaml file
        self.debug = False
        self.samplerate = 200
        self.threshold = .01
        self.emgdir = d[0]['dataworks']['folders'][getpass.getuser()]['emgrawdir']

    def file_browser(self, word):
        """function which helps to find files with certain content, specified in the input"""
        file = []
        os.chdir(self.emgdir)

        for f in glob.glob("*"):
            if word in f:
                file.append(f)

        return file

    def butter_lowpass(self, cutoff, fs, order=5):
        """ This functions generates a lowpass butter filter"""

        nyq = 0.5 * fs  # Nyquist frequeny is half the sampling frequency
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order):
        """Low-pass Butterworth filter at cutoff frequency with a certain order """

        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def butter_highpass(self, cutoff, fs, order=5):
        """ This functions generates a higpass butter filter"""
        nyq = 0.5 * fs  # Nyquist frequeny is half the sampling frequency
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def butter_highpass_filter(self, data, cutoff, fs, order):
        """High-pass Butterworth filter at cutoff frequency with a certain order """
        b, a = self.butter_highpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def getPSD(self, raw_emg_signal):
        ## TODO include an overlap in the pwelch function
        frequencies, psd = welch(raw_emg_signal, fs=self.samplerate,
                                 window='hanning', nperseg=256, detrend='constant',
                                 scaling="spectrum")
        return [psd, frequencies]


class EMGFeaturesTime:
    """Time Domain Features"""

    def __init__(self):
        # TODO insert a list of features to extract in the yaml file
        self.debug = False
        # self.wdir = d[0]['dataworks']['folders'][getpass.getuser()]['wdir']

    def getIEMG(self, raw_emg_signal):
        """ Sum of absolute values of the signal amplitude.
        IEMG = sum(|xi|) for i = 1 --> N

            Input: 		raw EMG Signal as list
            Output: 	integrated EMG
        """

        IEMG = np.sum([abs(x) for x in raw_emg_signal])
        return IEMG

    def getMAV(self, raw_emg_signal):
        """ Average of signal amplitude.
        MAV = 1/N * sum(|xi|) for i = 1 --> N

            Input: 		raw EMG Signal as list
            Output: 	Mean Absolute Value
        """

        MAV = 1 / len(raw_emg_signal) * np.sum([abs(x) for x in raw_emg_signal])
        return MAV

    def getMAV1(self, raw_emg_signal):
        """ Average of EMG signal amplitude, using modified version n°1

                IEMG = 1/N * sum(wi|xi|) for i = 1 --> N
                wi = {
                      1 if 0.25N <= i <= 0.75N,
                      0.5 otherwise
                      }

            Input:		raw EMG Signal as list
            Output: 	Mean Absolute Value
        """
        wIndexMin = int(0.25 * len(raw_emg_signal))
        wIndexMin = int(0.25 * len(raw_emg_signal))
        wIndexMax = int(0.75 * len(raw_emg_signal))
        absoluteSignal = [abs(x) for x in raw_emg_signal]
        IEMG = 0.5 * np.sum([x for x in absoluteSignal[0:wIndexMin]]) + np.sum(
            [x for x in absoluteSignal[wIndexMin:wIndexMax]]) + 0.5 * np.sum([x for x in absoluteSignal[wIndexMax:]])
        MAV1 = IEMG / len(raw_emg_signal)
        return MAV1

    def getMAV2(self, raw_emg_signal):
        """ Average of EMG signal amplitude, using modified version n°2
    
                IEMG = 1/N * sum(wi|xi|) for i = 1 --> N
                wi = {
                      1 if 0.25N <= i <= 0.75N,
                      4i/N if i < 0.25N
                      4(i-N)/N otherwise
                      }
    
            Input:		raw EMG Signal as list
            Output: 	Mean Absolute Value   
        """

        N = len(raw_emg_signal)
        wIndexMin = int(0.25 * N)  # get the index at 0.25N
        wIndexMax = int(0.75 * N)  # get the index at 0.75N

        temp = []  # create an empty list
        for i in range(0, wIndexMin):  # case 1: i < 0.25N
            x = abs(raw_emg_signal[i] * (4 * i / N))
            temp.append(x)

            for i in range(wIndexMin, wIndexMax + 1):  # case2: 0.25 <= i <= 0.75N
                x = abs(raw_emg_signal[i])
            temp.append(x)

            for i in range(wIndexMax + 1, N):  # case3; i > 0.75N
                x = abs(raw_emg_signal[i]) * (4 * (i - N) / N)
            temp.append(x)

        MAV2 = np.sum(temp) / N
        return MAV2

    def getSSI(self, raw_emg_signal):
        """ Summation of square values of EMG signal
    
                SSI = sum(xi**2) for i = 1 --> N
    
            Input: 		raw EMG Signal as list
            Output: 	Simple Square Integral
        """

        SSI = np.sum([x ** 2 for x in raw_emg_signal])
        return SSI

    def getVAR(self, raw_emg_signal):
        """ Summation of average square values of a variables deviation
    
                VAR = (1 / (N - 1)) * sum(xi**2) for i = 1 --> N
    
            Input: 		raw EMG Signal as list
            Output: 	Summation of the average square values
        """

        SSI = np.sum([x ** 2 for x in raw_emg_signal])
        N = len(raw_emg_signal)
        VAR = SSI * (1 / (N - 1))
        return VAR

    def getTM(self, raw_emg_signal, order):
        """ 
            Temporal Moment of order X of the EMG signal
    
                TM = (1 / N * sum(xi**order) for i = 1 --> N
    
            Input: 		raw EMG Signal as list, order as an int
            Output: 	Temporal moment of order
        """
        N = len(raw_emg_signal)
        TM = abs((1 / N) * np.sum([x ** order for x in raw_emg_signal]))
        return TM

    def getRMS(self, raw_emg_signal):
        """ Root mean square of signal
    
                RMS = (sqrt( (1 / N) * sum(xi**2))) for i = 1 --> N
    
            Input: 		raw EMG Signal as list
            Output: 	Root mean square of signal
        """
        N = len(raw_emg_signal)
        RMS = np.sqrt((1 / N) * np.sum([x ** 2 for x in raw_emg_signal]))
        return RMS

    def getLOG(self, raw_emg_signal):
        """ LOG is a feature that provides an estimate of the muscle contraction force.::
    
                LOG = e^((1/N) * sum(|xi|)) for x i = 1 --> N
    
            Input: 		raw EMG Signal
            Output: 	*LOG    
        """
        LOG = math.exp((1 / len(raw_emg_signal)) * sum([abs(x) for x in raw_emg_signal]))
        return LOG

    def getWL(self, raw_emg_signal):
        """ Waveform length of the EMG signal, as a measure of its complexity
    
                WL = sum(|x(i+1) - xi|) for i = 1 --> N-1
    
            Input: 			raw EMG Signal as list
            Output: 		wavelength of the signal
        """
        N = len(raw_emg_signal)
        temp = []
        for i in range(0, N - 1):
            temp.append(abs(raw_emg_signal[i + 1] - raw_emg_signal[i]))
        WL = sum(temp)
        return WL

    def getAAC(self, raw_emg_signal):
        """ Average amplitude change
    
                AAC = 1/N * sum(|x(i+1) - xi|) for i = 1 --> N-1
    
            Input		raw EMG Signal as list
            Output: 	Average amplitude change of the signal
        """
        N = len(raw_emg_signal)
        WL = self.getWL(raw_emg_signal)
        ACC = 1 / N * WL
        return ACC

    def getDASDV(self, raw_emg_signal):
        """ Standard deviation value of the the wavelength
    
                DASDV = sqrt( (1 / (N-1)) * sum((x[i+1] - x[i])**2 ) for i = 1 --> N - 1    
    
            Input: 		raw EMG Signal
            Output:		DASDV
        """

        N = len(raw_emg_signal)
        temp = []
        for i in range(0, N - 1):
            temp.append((raw_emg_signal[i + 1] - raw_emg_signal[i]) ** 2)
        DASDV = (1 / (N - 1)) * sum(temp)
        return DASDV

    def getAFB(self, raw_emg_signal, samplerate, windowSize):
        """ Amplitude at first burst. For details see:
            Du, S., & Vuskovic, M. (2004, November). Temporal vs. spectral approach to
            feature extraction from prehensile EMG signals. In Proceedings of the 2004
            IEEE International Conference on Information Reuse and Integration, 2004.
            IRI 2004. (pp. 344-350). IEEE.

            Input: 		raw_emg_signal as list, Fs [in Hz], windowSize [in ms]
            Output: 	Amplitude at first burst
        """
        squaredSignal = square(raw_emg_signal)  # squaring the signal
        windowSample = int((windowSize * 1000) / samplerate)  # get the number of samples for each window
        w = np.hamming(windowSample)
        # From: http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
        filteredSignal = np.convolve(w / w.sum(), squaredSignal, mode='valid')
        peak = peakutils.indexes(filteredSignal)[0]
        AFB = filteredSignal[peak]
        return AFB

    def getZC(self, raw_emg_signal, threshold):
        """ How many times does the signal crosses the 0 (+-threshold).::
    
                ZC = sum([sgn(x[i] X x[i+1]) intersecated |x[i] - x[i+1]| >= threshold]) for i = 1 --> N - 1
                sign(x) = {
                            1, if x >= threshold
                            0, otherwise
                        }
    
            * Input:
                * raw_emg_signal = EMG signal as list
                * threshold = threshold to use in order to avoid fluctuations caused by noise and low voltage fluctuations
            * Output:
                * ZC index       
    
            :param raw_emg_signal: the raw EMG signal
            :type raw_emg_signal: list
            :param threshold: value to sum / substract to the zero when evaluating the crossing.
            :type threshold: int
            :return: Number of times the signal crosses the 0 (+- threshold)
            :rtype: float
        """
        positive = (raw_emg_signal[0] > threshold)
        ZC = 0
        for x in raw_emg_signal[1:]:
            if (positive):
                if (x < 0 - threshold):
                    positive = False
                    ZC += 1
            else:
                if (x > 0 + threshold):
                    positive = True
                    ZC += 1
        return ZC

    def getMYOP(self, raw_emg_signal, threshold):
        """ The myopulse percentage rate (MYOP) is an average value of myopulse output.
            It is defined as one absolute value of the EMG signal exceed a pre-defined thershold value. ::
    
                MYOP = (1/N) * sum(|f(xi)|) for i = 1 --> N
                f(x) = {
                        1 if x >= threshold
                        0 otherwise
                }
    
            * Input:
                * raw_emg_signal = EMG signal as list
                * threshold = threshold to avoid fluctuations caused by noise and low voltage fluctuations
            * Output:
                * Myopulse percentage rate
    
            :param raw_emg_signal: the raw EMG signal
            :type raw_emg_signal: list
            :param threshold: value to sum / substract to the zero when evaluating the crossing.
            :type threshold: int
            :return: Myopulse percentage rate of the signal
            :rtype: float
        """
        N = len(raw_emg_signal)
        MYOP = len([1 for x in raw_emg_signal if abs(x) >= threshold]) / N
        return MYOP

    def getWAMP(self, raw_emg_signal, threshold):
        """ Wilson or Willison amplitude is a measure of frequency information.
            It is a number of time resulting from difference between the EMG signal of two adjoining segments, that exceed a threshold.::
    
                WAMP = sum( f(|x[i] - x[i+1]|)) for n = 1 --> n-1
                f(x){
                    1 if x >= threshold
                    0 otherwise
                }
    
            * Input:
                * raw_emg_signal = EMG signal as list
                * threshold = threshold to avoid fluctuations caused by noise and low voltage fluctuations
            * Output:
                * Wilson Amplitude value
    
            :param raw_emg_signal: the raw EMG signal
            :type raw_emg_signal: list
            :param threshold: value to sum / substract to the zero when evaluating the crossing.
            :type threshold: int
            :return: Willison amplitude 
            :rtype: float
        """

        N = len(raw_emg_signal)
        WAMP = 0
        for i in range(0, N - 1):
            x = raw_emg_signal[i] - raw_emg_signal[i + 1]
            if (x >= threshold):
                WAMP += 1
            return (WAMP)

    def getSSC(self, raw_emg_signal, threshold):
        """ Number of times the slope of the EMG signal changes sign.::
    
                SSC = sum(f( (x[i] - x[i-1]) X (x[i] - x[i+1]))) for i = 2 --> n-1
    
                f(x){
                    1 if x >= threshold
                    0 otherwise
                }
    
            * Input: 
                * raw EMG Signal
            * Output: 
                * number of Slope Changes
    
            :param raw_emg_signal: the raw EMG signal
            :type raw_emg_signal: list
            :param threshold: value to sum / substract to the zero when evaluating the crossing.
            :type threshold: int
            :return: Number of slope's sign changes
            :rtype: int
        """

        N = len(raw_emg_signal)
        SSC = 0
        for i in range(1, N - 1):
            a, b, c = [raw_emg_signal[i - 1], raw_emg_signal[i], raw_emg_signal[i + 1]]
            if (a + b + c >= threshold * 3):  # computed only if the 3 values are above the threshold
                if (a < b > c or a > b < c):  # if there's change in the slope
                    SSC += 1
        return SSC

    def getMAVSLPk(self, raw_emg_signal, nseg):
        """ Mean Absolute value slope is a modified versions of MAV feature.
    
            The MAVs of adiacent segments are determinated. ::
    
                MAVSLPk = MAV[k+1] - MAV[k]; k = 1,..,k+1
    
            * Input: 
                * raw EMG signal as list
                * nseg = number of segments to evaluate
    
            * Output: 
                 * list of MAVs
    
            :param raw_emg_signal: the raw EMG signal
            :type raw_emg_signal: list
            :param nseg: number of segments to evaluate
            :type nseg: int
            :return: Mean absolute slope value
            :rtype: float
        """
        N = len(raw_emg_signal)
        lenK = int(N / nseg)  # length of each segment to compute
        MAVSLPk = []
        for s in range(0, N, lenK):
            MAVSLPk.append(self.getMAV(raw_emg_signal[s:s + lenK]))
        return MAVSLPk

    def getHIST(self, raw_emg_signal, nseg, threshold):
        """ Histograms is an extension version of ZC and WAMP features. 
    
            * Input:
                * raw EMG Signal as list
                * nseg = number of segment to analyze
                * threshold = threshold to use to avoid DC fluctuations
    
            * Output:
                * get zc/wamp for each segment
    
            :param raw_emg_signal: the raw EMG signal
            :type raw_emg_signal: list
            :param nseg: number of segments to analyze
            :type nseg: int
            :param threshold: value to sum / substract to the zero when evaluating the crossing.
            :type threshold: int
            :return: Willison amplitude 
            :rtype: float
        """
        segmentLength = int(len(raw_emg_signal) / nseg)
        HIST = {}
        for seg in range(0, nseg):
            HIST[seg + 1] = {}
            thisSegment = raw_emg_signal[seg * segmentLength: (seg + 1) * segmentLength]
            HIST[seg + 1]["ZC"] = self, self.getZC(thisSegment, threshold)
            HIST[seg + 1]["WAMP"] = self, self.getWAMP(thisSegment, threshold)
        return HIST


# -------------------------------------------------------------------------------------------------------------------#

class EMGFeaturesFreq:
    """Frequency domain Features"""

    def __init__(self):
        with open('/media/storage/iPScnn/config.yaml', 'r') as f:
            d = yaml.load(f.read())

        # TODO insert a list of features to extract in the yaml file
        self.debug = False
        self.wdir = d[0]['dataworks']['folders'][getpass.getuser()]['wdir']

    def getMNF(self, raw_emg_power_spectrum, frequencies):
        """ Obtain the mean frequency of the EMG signal, evaluated as the sum of
            product of the EMG power spectrum and the frequency divided by total sum of the spectrum intensity::

                MNF = sum(fPj) / sum(Pj) for j = 1 -> M
                M = length of the frequency bin
                Pj = power at freqeuncy bin j
                fJ = frequency of the spectrum at frequency bin j

            * Input:
                * raw_emg_power_spectrum: PSD as list
                * frequencies: frequencies of the PSD spectrum as list
            * Output:
                * Mean Frequency of the PSD

            :param raw_emg_power_spectrum: power spectrum of the EMG signal
            :type raw_emg_power_spectrum: list
            :param frequencies: frequencies of the PSD
            :type frequencies: list
            :return: mean frequency of the EMG power spectrum
            :rtype: float
        """
        a = []
        for i in range(0, len(frequencies)):
            a.append(frequencies[i] * raw_emg_power_spectrum[i])
        b = sum(raw_emg_power_spectrum)
        MNF = sum(a) / b
        return (MNF)

    def getMDF(self, raw_emg_power_spectrum, frequencies):
        """ Obtain the Median Frequency of the PSD.

            MDF is a frequency at which the spectrum is divided into two regions with equal amplitude, in other words, MDF is half of TTP feature

            * Input:
                * raw EMG Power Spectrum
                * frequencies
            * Output:
                * Median Frequency  (Hz)

            :param raw_emg_power_spectrum: power spectrum of the EMG signal
            :type raw_emg_power_spectrum: list
            :param frequencies: frequencies of the PSD
            :type frequencies: list
            :return: median frequency of the EMG power spectrum
            :rtype: float
        """
        MDP = sum(raw_emg_power_spectrum) * (1 / 2)
        for i in range(1, len(raw_emg_power_spectrum)):
            if (sum(raw_emg_power_spectrum[0:i]) >= MDP):
                return (frequencies[i])

    def getPeakFrequency(self, raw_emg_power_spectrum, frequencies):
        """ Obtain the frequency at which the maximum peak occur

            * Input:
                * raw EMG Power Spectrum as list
                * frequencies as list
            * Output:
                * frequency in Hz

            :param raw_emg_power_spectrum: power spectrum of the EMG signal
            :type raw_emg_power_spectrum: list
            :param frequencies: frequencies of the PSD
            :type frequencies: list
            :return: peakfrequency of the EMG Power spectrum
            :rtype: float
        """
        peakFrequency = frequencies[np.argmax(raw_emg_power_spectrum)]
        return (peakFrequency)

    def getMNP(self, raw_emg_power_spectrum):
        """ This functions evaluate the mean power of the spectrum.::

                Mean Power = sum(Pj) / M, j = 1 --> M, M = len of the spectrum

            * Input:
                * EMG power spectrum
            * Output:
                * mean power

            :param raw_emg_power_spectrum: power spectrum of the EMG signal
            :type raw_emg_power_spectrum: list
            :param frequencies: frequencies of the PSD
            :type frequencies: list
            :return: mean power of the EMG power spectrum
            :rtype: float
        """

        MNP = np.mean(raw_emg_power_spectrum)
        return (MNP)

    def getTTP(self, raw_emg_power_spectrum):
        """ This functions evaluate the aggregate of the EMG power spectrum (aka Zero Spectral Moment)

            * Input:
                * raw EMG Power Spectrum
            * Output:
                * Total Power

            :param raw_emg_power_spectrum: power spectrum of the EMG signal
            :type raw_emg_power_spectrum: list
            :param frequencies: frequencies of the PSD
            :type frequencies: list
            :return: total power of the EMG power spectrum
            :rtype: float
        """

        TTP = sum(raw_emg_power_spectrum)
        return (TTP)

    def getSM(self, raw_emg_power_spectrum, frequencies, order):
        """ Get the spectral moment of a spectrum::

                SM = sum(fj*(Pj**order)), j = 1 --> M

            * Input:
                * raw EMG Power Spectrum
                * frequencies as list
                * order (int)
            * Output:
                * SM of order = order

            :param raw_emg_power_spectrum: power spectrum of the EMG signal
            :type raw_emg_power_spectrum: list
            :param frequencies: frequencies of the PSD
            :type frequencies: list
            :param order: order to the moment
            :type order: int
            :return: Spectral moment of order X of the EMG power spectrum
            :rtype: float
        """
        SMo = []
        for j in range(0, len(frequencies)):
            SMo.append(frequencies[j] * (raw_emg_power_spectrum[j] ** order))
        SMo = sum(SMo)
        return (SMo)

    def getFR(self, raw_emg_power_spectrum, frequencies, llc=20, ulc=50, lhc=50, uhc=80):
        """ This functions evaluate the frequency ratio of the power spectrum.

            Cut-off value can be decidec experimentally or from the MNF Feature See: Oskoei, M.A., Hu, H. (2006). GA-based feature subset selection for myoelectric classification.

            * Input:
                * raw EMG power spectrum as list,
                * frequencies as list,
                * llc = lower low cutoff
                * ulc = upper low cutoff
                * lhc = lower high cutoff
                * uhc = upper high cutoff
            * Output:
                * Frequency Ratio

            :param raw_emg_power_spectrum: power spectrum of the EMG signal
            :type raw_emg_power_spectrum: list
            :param frequencies: frequencies of the PSD
            :type frequencies: list
            :param llc: lower cutoff frequency for the low frequency components
            :type llc: float
            :param ulc: upper cutoff frequency for the low frequency components
            :type ulc: float
            :param lhc: lower cutoff frequency for the high frequency components
            :type lhc: float
            :param uhc: upper cutoff frequency for the high frequency components
            :type uhc: float
            :return: frequencies ratio of the EMG power spectrum
            :rtype: float
        """
        frequencies = list(frequencies)
        # First we check for the closest value into the frequency list to the cutoff frequencies
        llc = min(frequencies, key=lambda x: abs(x - llc))
        ulc = min(frequencies, key=lambda x: abs(x - ulc))
        lhc = min(frequencies, key=lambda x: abs(x - lhc))
        uhc = min(frequencies, key=lambda x: abs(x - uhc))

        LF = sum([P for P in raw_emg_power_spectrum[frequencies.index(llc):frequencies.index(ulc)]])
        HF = sum([P for P in raw_emg_power_spectrum[frequencies.index(lhc):frequencies.index(uhc)]])
        FR = LF / HF
        return (FR)

    def getPSR(self, raw_emg_power_spectrum, frequencies, n=20, fmin=10, fmax=500):
        """ This function computes the Power Spectrum Ratio of the signal, defined as:
            Ratio between the energy P0 which is nearby the maximum value of the EMG power spectrum and the energy P which is the whole energy of the EMG power spectrum

            * Input:
                * EMG power spectrum
                * frequencies as list
                * n = range around f0 to evaluate P0
                * fmin = min frequency
                * fmax = max frequency

            :param raw_emg_power_spectrum: power spectrum of the EMG signal
            :type raw_emg_power_spectrum: list
            :param frequencies: frequencies of the PSD
            :type frequencies: list
            :param n: range of frequencies around f0 to evaluate
            :type n: int
            :param fmin: min frequency to evaluate
            :type fmin: int
            :param fmax: lmaximum frequency to evaluate
            :type fmax: int
            :return: Power spectrum ratio of the EMG power spectrum
            :rtype: float
        """

        frequencies = list(frequencies)

        # The maximum peak and frequencies are evaluate using the getPeakFrequency functions
        # First we check for the closest value into the frequency list to the cutoff frequencies
        peakFrequency = self.getPeakFrequency(raw_emg_power_spectrum, frequencies)
        f0min = peakFrequency - n
        f0max = peakFrequency + n
        f0min = min(frequencies, key=lambda x: abs(x - f0min))
        f0max = min(frequencies, key=lambda x: abs(x - f0max))
        fmin = min(frequencies, key=lambda x: abs(x - fmin))
        fmax = min(frequencies, key=lambda x: abs(x - fmax))

        # here we evaluate P0 and P
        P0 = sum(raw_emg_power_spectrum[frequencies.index(f0min):frequencies.index(f0max)])
        P = sum(raw_emg_power_spectrum[frequencies.index(fmin):frequencies.index(fmax)])
        PSR = P0 / P

        return (PSR)

    def getVCF(self, SM0, SM1, SM2):
        """This function evaluate the variance of the central freuency of the PSD.::

                VCF = (1 / SM0)*sum(Pj*(fj - fc)**2),j = 1 --> M, = SM2 / SM0 - (SM1 /SM0) **2

            * Input:
                * SM0: spectral moment of order 0
                * SM1: spectral moment of order 1
                * SM2: spectral moment of order 0
            * Output:
                * Variance of Central frequency of the Power spectrum

            :param SM0: Spectral moment of order 0
            :type SM0: float
            :param SM1: Spectral moment of order 1
            :type SM1: float
            :param SM2: Spectral moment of order 2
            :type SM2: float
            :return: Variance of central frequency
            :rtype: float
        """
        VCF = (SM2 / SM0) - (SM1 / SM0) ** 2
        return (VCF)
