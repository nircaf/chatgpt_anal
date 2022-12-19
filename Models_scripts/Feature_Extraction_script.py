import numpy as np
#import pyedflib
from matplotlib import pyplot as plt
from nitime import utils
from nitime import algorithms as alg
from nitime.timeseries import TimeSeries
from nitime.viz import plot_tseries
import csv
import pywt
import scipy.stats as sp
from spectrum import *
from os import listdir
from os.path import isfile, join
import heapq

from scipy.signal import argrelextrema
from scipy import signal
import scipy.io as sio
from tkinter.filedialog import askopenfilename, askdirectory
from tkinter import *

names = ['Activity', 'Mobility', 'Complexity', 'Kurtosis', '2nd Difference Mean', '2nd Difference Max', 'Coeffiecient of Variation', 'Skewness', '1st Difference Mean', '1st Difference Max',
          'Wavelet Approximate Mean', 'Wavelet Approximate Std Deviation', 'Wavelet Detailed Mean', 'Wavelet Detailed Std Deviation', 'Wavelet Approximate Energy', 'Wavelet Detailed Energy',
          'Wavelet Approximate Entropy', 'Wavelet Detailed Entropy', 'Variance', 'Mean of Vertex to Vertex Slope', 'FFT Delta MaxPower','FFT Theta MaxPower', 'FFT Alpha MaxPower', 'FFT Beta MaxPower',
          'Autro Regressive Mode Order 3 Coefficients for each channel ->']

def hjorth(input, num_of_chnls_together=1):                                             # function for hjorth
    realinput = input
    hjorth_activity = np.zeros(len(realinput))
    hjorth_mobility = np.zeros(len(realinput))
    hjorth_diffmobility = np.zeros(len(realinput))
    hjorth_complexity = np.zeros(len(realinput))
    diff_input = np.diff(realinput)
    diff_diffinput = np.diff(diff_input)
    k = 0
    for j in realinput:
        hjorth_activity[k] = np.var(j)
        hjorth_mobility[k] = np.sqrt(np.var(diff_input[k])/hjorth_activity[k])
        hjorth_diffmobility[k] = np.sqrt(np.var(diff_diffinput[k])/np.var(diff_input[k]))
        hjorth_complexity[k] = hjorth_diffmobility[k]/hjorth_mobility[k]
        k = k+1
    return np.sum(hjorth_activity)/num_of_chnls_together, np.sum(hjorth_mobility)/num_of_chnls_together, np.sum(hjorth_complexity)/num_of_chnls_together                       #returning hjorth activity, hjorth mobility , hjorth complexity



def my_kurtosis(a, num_of_chnls_together=1):
    b = a # Extracting the data from the 14 channels
    output = np.zeros(len(b)) # Initializing the output array with zeros (length = 14)
    k = 0; # For counting the current row no.
    for i in b:
        mean_i = np.mean(i) # Saving the mean of array i
        std_i = np.std(i) # Saving the standard deviation of array i
        t = 0.0
        for j in i:
            t += (pow((j-mean_i)/std_i,4)-3)
        kurtosis_i = t/len(i) # Formula: (1/N)*(summation(x_i-mean)/standard_deviation)^4-3
        output[k] = kurtosis_i # Saving the kurtosis in the array created
        k +=1 # Updating the current row no.
    return np.sum(output)/num_of_chnls_together

##----------------------------------------- End Kurtosis Function ----------------------------##


##------------------------------------- Begin 2ndDiffMean(Absolute difference) Function ------##
##-------------------------- [ Input: 2D array (row: Channels, column: Data)] --------------- ##
##-------------------  -- [ Output: 1D array (2ndDiffMean values for each channel)] ----------##

def secDiffMean(a, num_of_chnls_together=1):
    b = a # Extracting the data of the 14 channels
    output = np.zeros(len(b)) # Initializing the output array with zeros (length = 14)
    temp1 = np.zeros(len(b[0])-1) # To store the 1st Diffs
    k = 0 # For counting the current row no.
    for i in b:
        t = 0.0
        for j in range(len(i)-1):
            temp1[j] = abs(i[j+1]-i[j]) # Obtaining the 1st Diffs
        for j in range(len(i)-2):
            t += abs(temp1[j+1]-temp1[j]) # Summing the 2nd Diffs
        output[k] = t/(len(i)-2) # Calculating the mean of the 2nd Diffs
        k +=1 # Updating the current row no.
    return np.sum(output)/num_of_chnls_together

##------------------------------------- End 2ndDiffMean Function----- -------------------------##


##------------------------------------- Begin 2ndDiffMax Function(Absolute difference) --------##
##-------------------------- [ Input: 2D array (row: Channels, column: Data)] -----------------##
##--------------------- [ Output: 1D array (2ndDiffMax values for each channel)] --------------##

def secDiffMax(a, num_of_chnls_together=1):
    b = a # Extracting the data from the 14 channels
    output = np.zeros(len(b)) # Initializing the output array with zeros (length = 14)
    temp1 = np.zeros(len(b[0])-1) # To store the 1st Diffs
    k = 0; # For counting the current row no.
    t = 0.0
    for i in b:
        for j in range(len(i)-1):
            temp1[j] = abs(i[j+1]-i[j]) # Obtaining the 1st Diffs
        t = temp1[1] - temp1[0]
        for j in range(len(i)-2):
            if abs(temp1[j+1]-temp1[j]) > t:
                t = temp1[j+1]-temp1[j] # Comparing current Diff with the last updated Diff Max

        output[k] = t # Storing the 2nd Diff Max for channel k
        k +=1 # Updating the current row no.
    return np.sum(output)/num_of_chnls_together



def wrapper1(a):
    kurtosis =  my_kurtosis(a)
    sec_diff_mean = secDiffMean(a)
    sec_diff_max  = secDiffMax(a)
    return kurtosis, sec_diff_mean, sec_diff_max



def coeff_var(a, num_of_chnls_together=1):
    b = a #Extracting the data from the 14 channels
    output = np.zeros(len(b)) #Initializing the output array with zeros
    k = 0 #For counting the current row no.
    for i in b:
        mean_i = np.mean(i) #Saving the mean of array i
        std_i = np.std(i) #Saving the standard deviation of array i
        output[k] = std_i/mean_i #computing coefficient of variation
        k = k+1
    return np.sum(output)/num_of_chnls_together


def skewness(arr, num_of_chnls_together=1):
    data = arr
    skew_array = np.zeros(len(data))  # Initialinling the array as all 0s
    index = 0  # current cell position in the output array

    for i in data:
        skew_array[index] = sp.stats.skew(i, axis=0, bias=True)
        index += 1  # updating the cell position
    return np.sum(skew_array) / num_of_chnls_together


def first_diff_mean(arr, num_of_chnls_together=1):
    data = arr
    diff_mean_array = np.zeros(len(data))  # Initialinling the array as all 0s
    index = 0  # current cell position in the output array

    for i in data:
        sum = 0.0  # initializing the sum at the start of each iteration
        for j in range(len(i) - 1):
            sum += abs(i[j + 1] - i[j])  # Obtaining the 1st Diffs

        diff_mean_array[index] = sum / (len(i) - 1)
        index += 1  # updating the cell position
    return np.sum(diff_mean_array) / num_of_chnls_together


def first_diff_max(arr, num_of_chnls_together=1):
    data = arr
    diff_max_array = np.zeros(len(data))  # Initialinling the array as all 0s
    first_diff = np.zeros(len(data[0]) - 1)  # Initialinling the array as all 0s
    index = 0  # current cell position in the output array

    for i in data:
        max = 0.0  # initializing at the start of each iteration
        for j in range(len(i) - 1):
            first_diff[j] = abs(i[j + 1] - i[j])  # Obtaining the 1st Diffs
            if first_diff[j] > max:
                max = first_diff[j]  # finding the maximum of the first differences
        diff_max_array[index] = max
        index += 1  # updating the cell position
    return np.sum(diff_max_array) / num_of_chnls_together


def wrapper2(arr):
    skew = skewness(arr)
    fdmean = first_diff_mean(arr)
    fdmax = first_diff_max(arr)
    return skew, fdmean, fdmax


def wavelet_features(epoch, num_of_chnls_together=1):
    cA_values = []
    cD_values = []
    cA_mean = []
    cA_std = []
    cA_Energy =[]
    cD_mean = []
    cD_std = []
    cD_Energy = []
    Entropy_D = []
    Entropy_A = []
    for i in range(num_of_chnls_together):
        eps = 1e-6
        cA,cD=pywt.dwt(epoch[i, :], 'coif1')
        cA_values.append(cA + eps)
        cD_values.append(cD + eps)		#calculating the coefficients of wavelet transform.
    for x in range(num_of_chnls_together):
        cA_mean.append(np.mean(cA_values[x]))
        cA_std.append(np.std(cA_values[x]))
        cA_Energy.append(np.sum(np.square(cA_values[x])))
        cD_mean.append(np.mean(cD_values[x]))		# mean and standard deviation values of coefficents of each channel is stored .
        cD_std.append(np.std(cD_values[x]))
        cD_Energy.append(np.sum(np.square(cD_values[x])))
        Entropy_D.append(np.sum(np.square(cD_values[x]) * np.log(np.square(cD_values[x]))))
        Entropy_A.append(np.sum(np.square(cA_values[x]) * np.log(np.square(cA_values[x]))))
    return np.sum(cA_mean)/num_of_chnls_together, np.sum(cA_std)/num_of_chnls_together, np.sum(cD_mean)/num_of_chnls_together, np.sum(cD_std)/num_of_chnls_together, np.sum(cA_Energy)/num_of_chnls_together, np.sum(cD_Energy)/num_of_chnls_together, np.sum(Entropy_A)/num_of_chnls_together, np.sum(Entropy_D)/num_of_chnls_together


def first_diff(i):
    b = i

    out = np.zeros(len(b))

    for j in range(len(i)):
        out[j] = b[j - 1] - b[j]  # Obtaining the 1st Diffs

        j = j + 1
        c = out[1:len(out)]
    return c


# first_diff(s)

def slope_mean(p, num_of_chnls_together=1):
    b = p  # Extracting the data from the 14 channels
    output = np.zeros(len(b))  # Initializing the output array with zeros
    res = np.zeros(len(b) - 1)

    k = 0  # For counting the current row no.
    for i in b:
        x = i
        amp_max = i[argrelextrema(x, np.greater)[0]]
        t_max = argrelextrema(x, np.greater)[0]
        amp_min = i[argrelextrema(x, np.less)[0]]
        t_min = argrelextrema(x, np.less)[0]
        t = np.concatenate((t_max, t_min), axis=0)
        t.sort()  # sort on the basis of time

        h = 0
        amp = np.zeros(len(t))
        try:
            res = np.zeros(len(t) - 1)
        except:
            return 0
        for l in range(len(t)):
            amp[l] = i[t[l]]

        amp_diff = first_diff(amp)

        t_diff = first_diff(t)

        for q in range(len(amp_diff)):
            res[q] = amp_diff[q] / t_diff[q]
        output[k] = np.mean(res)
        k = k + 1
    return np.sum(output) / num_of_chnls_together


def first_diff(i):
    b = i

    out = np.zeros(len(b))

    for j in range(len(i)):
        out[j] = b[j - 1] - b[j]  # Obtaining the 1st Diffs

        j = j + 1
        c = out[1:len(out)]
    return c  # returns first diff


def slope_var(p, num_of_chnls_together=1):
    b = p  # Extracting the data from the 14 channels
    output = np.zeros(len(b))  # Initializing the output array with zeros
    res = np.zeros(len(b) - 1)

    k = 0  # For counting the current row no.
    for i in b:
        x = i
        amp_max = i[argrelextrema(x, np.greater)[0]]  # storing maxima value
        t_max = argrelextrema(x, np.greater)[0]  # storing time for maxima
        amp_min = i[argrelextrema(x, np.less)[0]]  # storing minima value
        t_min = argrelextrema(x, np.less)[0]  # storing time for minima value
        t = np.concatenate((t_max, t_min), axis=0)  # making a single matrix of all matrix
        t.sort()  # sorting according to time

        h = 0
        amp = np.zeros(len(t))
        try:
            res = np.zeros(len(t) - 1)
        except:
            return 0
        for l in range(len(t)):
            amp[l] = i[t[l]]

        amp_diff = first_diff(amp)

        t_diff = first_diff(t)

        for q in range(len(amp_diff)):
            res[q] = amp_diff[q] / t_diff[q]  # calculating slope

        output[k] = np.var(res)
        k = k + 1  # counting k
    return np.sum(output) / num_of_chnls_together


def wrapper3(epoch):
    var1 = slope_mean(epoch)
    var2 = slope_var(epoch)
    return var1, var2


def maxPwelch(data_win, Fs, num_of_chnls_together=1):
    BandF = [0.1, 3, 7, 12, 30]
    PMax = np.zeros([14, (len(BandF) - 1)]);

    for j in range(num_of_chnls_together):
        f, Psd = signal.welch(data_win[j, :], Fs)

        for i in range(len(BandF) - 1):
            fr = np.where((f > BandF[i]) & (f <= BandF[i + 1]))
            PMax[j, i] = np.max(Psd[fr])

    return np.sum(PMax[:, 0]) / num_of_chnls_together, np.sum(PMax[:, 1]) / num_of_chnls_together, np.sum(PMax[:, 2]) / num_of_chnls_together, np.sum(PMax[:, 3]) / num_of_chnls_together


def entropy(labels): # Shanon Entropy
    """ Computes entropy of 0-1 vector. """
    n_labels = len(labels)
    counts = np.bincount(labels)
    probs = counts[np.nonzero(counts)] / n_labels
    n_classes = len(probs)

    if n_classes <= 1:
        return 0
    return - np.sum(probs * np.log(probs)) / np.log(n_classes)


def autogressiveModelParameters(labels, num_of_chnls_together=1):
    b_labels = len(labels)
    feature = []
    for i in range(num_of_chnls_together):
        coeff, sig = alg.AR_est_YW(labels[i, :], 11, )
        feature.append(coeff)
    a = []
    for i in range(11):
        a.append(np.sum(feature[:][i]) / num_of_chnls_together)

    return a


def autogressiveModelParametersBurg(labels, num_of_chnls_together=1):
    flag = 0
    feature = []
    feature1 = []
    model_order = 3
    for i in range(num_of_chnls_together):
        try:
            AR, rho, ref = arburg(labels[i], model_order)
            feature.append(AR)
        except:
            feature.append(0)
            flag = 1
    for j in range(num_of_chnls_together):
        for i in range(model_order):
            try:
                feature1.append(feature[j][i])
            except:
                feature1.append(0)

    return feature1, flag

def FE(path = None, create_mode=None):
    lowfiles = [f for f in listdir('C:\\Users\\Avishay David Malka\\Downloads\\Feature-Extraction-EEG-master\\Training-Data\\Low') if isfile(join('C:\\Users\\Avishay David Malka\\Downloads\\Feature-Extraction-EEG-master\\Training-Data\\Low', f))]
    highfiles = [f for f in listdir('C:\\Users\\Avishay David Malka\\Downloads\\Feature-Extraction-EEG-master\\Training-Data\\High') if isfile(join('C:\\Users\\Avishay David Malka\\Downloads\\Feature-Extraction-EEG-master\\Training-Data\\High', f))]
    files = []

    for i in lowfiles:
        files.append([i, 'Low'])

    for i in highfiles:
        files.append([i, 'High'])

    mypath = 'C:\\Users\\Avishay David Malka\\Downloads\\Feature-Extraction-EEG-master\\Training-Data\\'
    csvfile = "C:\\Users\\Avishay David Malka\\Downloads\\Feature-Extraction-EEG-master\\Features\\features.csv"
    if path is None:
        path = askopenfilename(title='Select the Brain MNIST object file',
                                      initialdir=os.path.abspath(
                                          'C:\\Users\\Avishay David Malka\\Work\\AI_dsp\\Data\\Raw data\\Sample Data_Brain_MNIST'))
    if create_mode is None:
        main_object = sio.loadmat(path)
        Raw_features = main_object['Features_raw']
        num_of_chnls_togheter = 1
        num_of_seq = 5
        seq_len = Raw_features.shape[2] // num_of_seq
        create_mode = 'all'
    elif create_mode is not None:
        create_mode = 'just_norm'
        csvfile = path
    cnt = 0
    if create_mode == 'all':
        with open(csvfile, "a") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerow(names)
            for counter in range(Raw_features.shape[0]):
                # subfolder = files[counter][1]
                # tag = files[counter][1]
                # data_path = mypath + subfolder + '/' + files[counter][0]
                # f = pyedflib.EdfReader(data_path)
                # n = f.signals_in_file
                # signal_labels = f.getSignalLabels()
                # sigbufs = np.zeros((num_of_chnls_togheter, seq_len))
                Raw_features_ud = Raw_features[:, :, 0:num_of_seq*seq_len]
                sigbufs = np.zeros((num_of_chnls_togheter, Raw_features_ud.shape[2]))
                for ch in range(int(Raw_features.shape[1] / num_of_chnls_togheter)):
                    for i in np.arange(num_of_chnls_togheter):
                        sigbufs[i, :] = Raw_features_ud[counter, (ch * num_of_chnls_togheter):((ch + 1) * num_of_chnls_togheter), :]
                    for i in range(num_of_seq):
                        features = []
                        epoch = sigbufs[:, (i * seq_len):((i + 1) * seq_len)]
                        if len(epoch[0]) == 0:
                            break

                        # Hjorth Parameters
                        feature_list = hjorth(epoch)
                        for feat in feature_list:
                            features.append(feat)

                        # Kurtosis , 2nd Diff Mean, 2nd Diff Max
                        feature_list = wrapper1(epoch)
                        for feat in feature_list:
                            features.append(feat)

                        # Coeffeicient of Variation
                        feat = coeff_var(epoch)
                        features.append(feat)

                        # Skewness , 1st Difference Mean, 1st Difference Max
                        feature_list = wrapper2(epoch)
                        for feat in feature_list:
                            features.append(feat)

                        # wavelet transform features
                        feature_list = wavelet_features(epoch)
                        for feat in feature_list:
                            features.append(feat)

                        # Variance and mean of Vertex to Vertex Slope
                        feature_list = wrapper3(epoch)
                        for feat in feature_list:
                            features.append(feat)

                        # Fast Fourier Transform features(Max Power)
                        feature_list = maxPwelch(epoch, 128)
                        for feat in feature_list:
                            features.append(feat)

                        # Autoregressive model Coefficients
                        feature_list, flag = autogressiveModelParametersBurg(epoch)
                        if flag:
                            continue
                        for feat in feature_list:
                            features.append(feat.real)
                        # check = np.asarray(features)
                        if (np.isnan(np.asarray(features))).any():
                            continue
                        writer.writerow(features)
                        cnt += 1

                if counter % int(Raw_features.shape[0] / 10) == 0:
                    print(f'Until now it was done {100*counter//Raw_features.shape[0]}% of the feature extraction process')
        with open(csvfile, "r") as output:
            r = csv.reader(output)  # Here your csv file
            lines = [l for l in r]

            for i in range(len(lines[1]) - 1):
                columns = []
                for j in range(1, len(lines)):
                    columns.append(float(lines[j][i]))
                mean = np.mean(columns, axis=0)
                std_dev = np.std(columns, axis=0)

                for j in range(1, len(lines)):
                    lines[j][i] = (float(lines[j][i]) - mean) / std_dev
            with open('C:\\Users\\Avishay David Malka\\Downloads\\Feature-Extraction-EEG-master\\Features\\Normalizedfeatures.csv', 'a') as norm_output:
                writer_norm = csv.writer(norm_output, lineterminator='\n')

                      # This file will store the normalized features
                writer_norm.writerows(lines)
    elif create_mode == 'just_norm':
        with open(csvfile, "r") as output:
            r = csv.reader(output)  # Here your csv file
            lines = [l for l in r]
            DS = np.asarray(lines[1::])

    try:
        return DS
    except:
        pass


if __name__ == '__main__':
    object_path = askopenfilename(title='Select the Brain MNIST object file',
                                  initialdir=os.path.abspath(
                                      'C:\\Users\\Avishay David Malka\\Work\\AI_dsp\\Data\\Raw data\\Sample Data_Brain_MNIST'))
    FE(path=object_path)
