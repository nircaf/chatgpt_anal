import glob
import matplotlib.pyplot as plt

import torch

import numpy as np
import numpy as np
from scipy.fftpack import dct
import mne
import re

def eeg_dft_generator(eeg_recording, window_size=1.0):
    # Convert the window size from seconds to the number of samples in the recording
    window_size_samples = int(window_size * eeg_recording.sampling_rate)
    
    # Iterate over the recording, yielding a window of samples and their DFT at each iteration
    for start in range(0, len(eeg_recording), window_size_samples):
        window = eeg_recording[start:start + window_size_samples]
        dft = np.fft.fft(window)
        yield dft


def eeg_dft_array(eeg_recording, window_size=1.0):
    # Convert the window size from seconds to the number of samples in the recording
    window_size_samples = int(window_size * eeg_recording.info['sfreq'])
    batch_size = eeg_recording._raw_lengths[0] // window_size_samples
    # Initialize the array that will store the DFT of the recording
    num_channels = eeg_recording.info['nchan']
    dft_array = np.zeros((window_size_samples, num_channels, batch_size), dtype=np.complex128)
    raw_data = eeg_recording.get_data()
    # Iterate over the recording, computing the DFT of each window and storing it in the array
    for start in range(0, len(raw_data), window_size_samples * batch_size):
        for i in range(batch_size):
            window = raw_data[:,start + i * window_size_samples: start + (i + 1) * window_size_samples]
            for j in range(num_channels):
                dft_array[:, j, i] = np.fft.fft(window[j, :])
    return dft_array


class EEGDFTClassifier(torch.nn.Module):
    def __init__(self, num_channels, window_size):
        super().__init__()
        
        # Create the layers of the model
        self.fc1 = torch.nn.Linear(num_channels * window_size, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 2)
        
    def forward(self, x):
        # Flatten the input tensor
        x = x.flatten(start_dim=1)
        
        # Apply the layers of the model
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        
        return x
    
def classify_eeg_dft(dft_array, device):
    # Convert the input array to a PyTorch tensor
    dft_tensor = torch.from_numpy(dft_array).to(torch.float32).to(device)
    
    # Create the classifier model and move it to the specified device
    num_channels, window_size, batch_size = dft_array.shape

    model = EEGDFTClassifier(num_channels, window_size).to(device)
    
    # Use the model to classify the input tensor
    with torch.no_grad():
        for i in range(batch_size):
            output = model(dft_tensor[:, :, i])
        
    # Return the predicted labels for each batch of samples
    _, labels = torch.max(output, dim=1)
    return labels


def process_and_plot_eeg_files(directory, device):
    # Use glob to find all the .edf files in the directory and its subdirectories
    file_pattern = f'{directory}/**/*.edf'
    filenames = glob.glob(file_pattern, recursive=True)
    
    # Load each .edf file and process it using the eeg_dft_array and classify_eeg_dft functions
    for filename in filenames:
        eeg_recording = mne.io.read_raw_edf(filename)
        dft_array = eeg_dft_array(eeg_recording)
        labels = classify_eeg_dft(dft_array, device)
        
        # Plot the EEG recording for each channel, using the predicted labels to color the data points
        time = np.arange(0, dft_array.shape[-1]) / eeg_recording.sampling_rate
        num_channels = eeg_recording.num_channels
        for i in range(num_channels):
            plt.plot(time, eeg_recording[:, i], color=labels[i])
        
        # Add labels and a legend to the plot
        plt.xlabel('Time (s)')
        plt.ylabel('EEG recording')
        plt.legend(['Channel 1', 'Channel 2', ...])
        
        # Show the plot
        plt.show()

def get_labels(path):
    # get .txt files
    txt_files = glob.glob(path + '/*.txt')
    dict_channels,dict_seizures = read_txt_file(txt_files[0])
    return dict_channels,dict_seizures


def read_txt_file(filename):
    # Initialize the dictionaries
    channels = {}
    seizures = {}
    
    # Compile the regular expressions
    channel_pattern = re.compile(r'Channel (\d+): (.*)      \n')
    seizure_pattern = re.compile(r'Seizure n (\d+)')
    filename_pattern = re.compile(r'File name: (.*)')
    times_pattern = re.compile(r'.*: +(\d{2}.\d{2}.\d{2})')

    # Read the file line by line
    with open(filename) as f:
        lines = f.readlines()
        
    # Parse the lines to extract the channel and seizure information
    for line in lines:
        channel_match = channel_pattern.match(line)
        if channel_match:
            # Parse a line with channel information
            channel_number = int(channel_match.group(1))
            channel_name = channel_match.group(2)
            channels[channel_number] = channel_name
            
        seizure_match = seizure_pattern.match(line)
        if seizure_match:
            # Parse a line with seizure information
            seizure_number = int(seizure_match.group(1))
            seizures[seizure_number] = {}
            
        filename_match = filename_pattern.match(line)
        if filename_match:
            # Parse a line with the filename
            seizures[seizure_number]['filename'] = filename_match.group(1)
            
        times_match = times_pattern.match(line)
        if times_match:
            # Parse a line with the registration and seizure times
            seizures[seizure_number][line.split(':')[0]] = times_match.group(1)
            
    return channels, seizures

if __name__ == '__main__':
    # Use the GPU if it's available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    import os

    # Set the directory where you want to start the search
    start_dir = "EEG_neurofeedback"

    # Loop through all directories and subdirectories in the start directory
    for root, dirs, files in os.walk(start_dir):
        # Check if the current directory contains both a .edf and .txt file
        if any(file.endswith(".edf") for file in files) and any(file.endswith(".txt") for file in files):
            # Get the labels for the EEG recordings
            get_labels(root)
            # Process and plot the EEG recordings in the data directory
            process_and_plot_eeg_files(dirs, device)