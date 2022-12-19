import lgbm
from chatgpt_anal import *
import os
import glob,mne
from torch.utils.data import Dataset, DataLoader,random_split
import sklearn
import pandas as pd
import kerasmodels
import pyod_ano_det
import vmdpy_run
import eegnet
import examples_deap_ccnn
from Models_scripts.from_1D_to_3D_pipeline_NN import *

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
                    # x_data = (T,C)
                    x_data = pd.concat([x_data,raw_recording.T])
                    y_data = pd.concat([y_data,pd.DataFrame(labels)])
                    # run over x_data columns
                    # for col in x_data.columns:
                    #     vmdpy_run.main(x_data[col], T=x_data.shape[0],fs=eeg_recording.info['sfreq'])

    # remove cols that all nan in x_data
    x_data = x_data.dropna(axis=1, how='any')
    # % of sum of y data
    print(100*y_data.sum()/len(y_data))
    models = [SleepStagerChambon2018,SleepStagerChambon2018_domain_adaptation,SleepStagerChambon2018_transfer_learning,SleepStagerChambon2018_fusion,
    SleepStagerChambon2018_super_unsuper_vised,SleepStagerChambon2018_with_unsupervides,SleepStagerChambon2018_deeper,SleepStagerChambon2018_with_gru,
    SleepStagerChambon2018_with_esn,SleepStagerChambon2018_with_esn_EchoTorch,SleepStagerChambon2018_ud,SleepStagerChambon2018_regression,SleepStagerChambon2018_UD]
    for model_run in models:
        model = model_run(x_data.shape[1], eeg_recording.info['sfreq'], n_classes=1)
        run_torch_model(model,x_data,y_data)
    # sklearn train test split
    data_train, data_test, target_train, target_test= sklearn.model_selection.train_test_split(x_data, y_data, test_size=0.2, random_state=1)
    data_train, data_val, target_train, target_val = sklearn.model_selection.train_test_split(data_train, target_train, test_size=0.25, random_state=1)
    dataset_trian = EEGDataset(data_train,target_train)
    dataset_val = EEGDataset(data_val,target_val)
    dataset_test = EEGDataset(data_test,target_test)
    # examples_deap_ccnn.run(EEGDataset(x_data,y_data))
    # kerasmodels.ResNet(x_data)
    # kerasmodels.run_keras(x_data, y_data, channelnum=45, sf=521, nb_classes=1)
    # train with torch
    # dft_array_dataset = prep_seizures_data_labels(x_data.T, y_data)
    # classify_eeg_dft(dft_array_dataset, eeg_recording.info['nchan'], eeg_recording.info['sfreq'])

    eegnet.eeg_net_run(eeg_recording, data_train,target_train,data_val,target_val,data_test,target_test)
    anomaly_detection(x_data.to_numpy())
    pyod_ano_det.main(data_train, data_test, target_train, target_test)
    lgbm.ensemble_model(data_train, data_test, target_train, target_test)


if __name__ == '__main__':
    # call the main function
    run()
