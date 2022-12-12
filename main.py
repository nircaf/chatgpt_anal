import lgbm
from chatgpt_anal import *
import os
import glob,mne
from torch.utils.data import Dataset, DataLoader,random_split
import sklearn
import pandas as pd
import kerasmodels
def run():
    # Set the directory where you want to start the search
    start_dir = "EEG_neurofeedback"
    org_func = eeg_dft_array

    if org_func == eeg_dft_array:
        x_data = pd.DataFrame()
        y_data = pd.DataFrame()
    else:
        x_data = np.array([]).reshape(0,0,0)
        y_data = np.empty((1))
    # Loop through all directories and subdirectories in the start directory
    for root, dirs, files in os.walk(start_dir):
        # Check if the current directory contains both a .edf and .txt file
        if any(file.endswith(".edf") for file in files) and any(file.endswith(".txt") for file in files):
            # Get the labels for the EEG recordings
            channels, seizures = get_labels(root)
            # Use glob to find all the .edf files in the directory and its subdirectories
            file_pattern = f'{root}/*.edf'
            filenames = glob.glob(file_pattern, recursive=True)

            # Load each .edf file and process it using the eeg_dft_array and classify_eeg_dft functions
            for filename in filenames:
                eeg_recording = mne.io.read_raw_edf(filename)
                raw_recording,labels = org_func(eeg_recording,seizures) # eeg_dft_array
                if org_func == eeg_dft_array:
                    x_data = pd.concat([x_data,raw_recording.T])
                    y_data = pd.concat([y_data,pd.DataFrame(labels)])
                else:
                    x_data = np.vstack((x_data,raw_recording.T))
                    y_data = np.concatenate((y_data,np.array(labels)))
    # remove cols that all nan in x_data
    x_data = x_data.dropna(axis=1, how='any')
    # kerasmodels.run_keras(x_data, y_data, channelnum=45, sf=521, nb_classes=1)
    # train with torch
    dft_array_dataset = prep_seizures_data_labels(x_data.T, y_data)
    # unsupervised_eeg(dft_array_dataset, eeg_recording.info['nchan'], eeg_recording.info['sfreq'])
    classify_eeg_dft(dft_array_dataset, eeg_recording.info['nchan'], eeg_recording.info['sfreq'])
    # sklearn train test split
    data_train, data_test, target_train, target_test = sklearn.model_selection.train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    lgbm.ensemble_model(data_train, data_test, target_train, target_test)

if __name__ == '__main__':
    # call the main function
    run()
