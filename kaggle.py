import tensorflow.compat.v1 as tf
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.io import loadmat
import os
from pywt import wavedec
from functools import reduce
from scipy import signal
from scipy.stats import entropy
from scipy.fft import fft, ifft
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow import keras as K
import matplotlib.pyplot as plt
import scipy
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold,cross_validate
from tensorflow.keras.layers import Dense, Activation, Flatten, concatenate, Input, Dropout, LSTM, Bidirectional,BatchNormalization,PReLU,ReLU,Reshape
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential, Model, load_model
import matplotlib.pyplot as plt;
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.decomposition import PCA
from tensorflow import keras
from sklearn.model_selection import cross_val_score
from tensorflow.keras.layers import Conv1D,Conv2D,Add
from tensorflow.keras.layers import MaxPool1D, MaxPooling2D
import seaborn as sns
import glob,mne
import warnings
warnings.filterwarnings('ignore')

from chatgpt_analysis import *


# Use glob to find all the .edf files in the directory and its subdirectories
root = 'EEG_neurofeedback'
file_pattern = f'{root}/*.edf'
filenames = glob.glob(file_pattern, recursive=True)
channels, seizures = get_labels(root)

# Load each .edf file and process it using the eeg_dft_array and classify_eeg_dft functions
for filename in filenames:
    eeg_recording = mne.io.read_raw_edf(filename)
    labels = np.zeros(( batch_size), dtype=np.float64)
    dft_array_dataset = eeg_dft_array(eeg_recording,seizures)

    x_train,x_test,y_train,y_test = train_test_split(eeg_recording.get_data(),y,test_size=0.15)
