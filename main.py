import lgbm
from chatgpt_anal import eeg_dft_array,get_labels,eeg_dft_array_w_fft
import os
import glob,mne
from torch.utils.data import Dataset, DataLoader,random_split
import sklearn
def run():
    # Set the directory where you want to start the search
    start_dir = "EEG_neurofeedback"

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
                raw_recording,labels = eeg_dft_array(eeg_recording,seizures)
                raw_recording = raw_recording.T
                # sklearn train test split
                data_train, data_test, target_train, target_test = sklearn.model_selection.train_test_split(raw_recording, labels, test_size=0.2, random_state=42)
                lgbm.ensemble_model(data_train, data_test, target_train, target_test)

if __name__ == '__main__':
    # call the main function
    run()
