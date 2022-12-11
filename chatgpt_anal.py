import glob
import matplotlib.pyplot as plt

import torch
import numpy as np
import numpy as np
from scipy.fftpack import dct
import mne
import re
from torch.utils.data import Dataset, DataLoader,random_split
import torch.nn as nn
import torch.nn.functional as F
def eeg_dft_generator(eeg_recording, window_size=1.0):
    # Convert the window size from seconds to the number of samples in the recording
    window_size_samples = int(window_size * eeg_recording.sampling_rate)

    # Iterate over the recording, yielding a window of samples and their DFT at each iteration
    for start in range(0, len(eeg_recording), window_size_samples):
        window = eeg_recording[start:start + window_size_samples]
        dft = np.fft.fft(window)
        yield dft

def seizures_start_time(seizures,dict_key='Seizure start time'):
    arr = []
    for i in seizures:
        arr.append(seizures[i][dict_key])
    return np.array(arr)

def eeg_dft_array_w_fft(eeg_recording,seizures, window_size=1.0):
    # Convert the window size from seconds to the number of samples in the recording
    window_size_samples = int(window_size * eeg_recording.info['sfreq'])
    batch_size = eeg_recording._raw_lengths[0] // window_size_samples
    # Initialize the array that will store the DFT of the recording
    num_channels = eeg_recording.info['nchan']
    dft_array = np.zeros((batch_size, num_channels,window_size_samples), dtype=np.float64)
    labels = np.zeros(( batch_size), dtype=np.int64)
    # get the raw data
    raw_data = eeg_recording.get_data()
    # seizure start and end time array
    seizures_start_time_arr = seizures_start_time(seizures)
    seizures_end_time_arr = seizures_start_time(seizures,dict_key='Seizure end time')
    # Iterate over the recording, computing the DFT of each window and storing it in the array
    for start in range(0, len(raw_data), window_size_samples * batch_size):
        for i in range(batch_size):
            start_sec = (start + i * window_size_samples) /eeg_recording.info['sfreq']
            for num_seizures in range(len(seizures_start_time_arr)):
                # if start sec > seizure start time and start sec < seizure end time for that seizure
                if start_sec >= seizures_start_time_arr[num_seizures] and start_sec <= seizures_end_time_arr[num_seizures]:
                    labels[i] = 1
                    break
            window = raw_data[:,start + i * window_size_samples: start + (i + 1) * window_size_samples]
            for j in range(num_channels):
                dft_array[i,j,:] = window[j, :]
                # dft_array[i,j,:] = np.fft.fft(window[j, :])
    # unsqueeze the dft_array to add a channel dimension
    dft_array_dataset = EEGDataset(dft_array, labels)
    return dft_array, labels

def eeg_dft_array(eeg_recording,seizures, window_size=1.0):
    # Convert the window size from seconds to the number of samples in the recording
    window_size_samples = int(window_size * eeg_recording.info['sfreq'])
    batch_size = eeg_recording._raw_lengths[0] // window_size_samples
    # Initialize the array that will store the DFT of the recording
    num_channels = eeg_recording.info['nchan']
    dft_array = np.zeros((batch_size, num_channels,window_size_samples), dtype=np.float64)
    # get the raw data
    raw_data = eeg_recording.get_data()
    labels = np.zeros(raw_data.shape[1], dtype=np.int64)

    # seizure start and end time array
    seizures_start_time_arr = seizures_start_time(seizures)*eeg_recording.info['sfreq']
    seizures_end_time_arr = seizures_start_time(seizures,dict_key='Seizure end time')*eeg_recording.info['sfreq']
    # run over labels and set 1 if there is a seizure
    for i in range(raw_data.shape[1]):
        for seizure in range(len(seizures_start_time_arr)):
            if i >= seizures_start_time_arr[seizure] and i <= seizures_end_time_arr[seizure]:
                labels[i] = 1
                break
    # unsqueeze the dft_array to add a channel dimension
    dft_array_dataset = EEGDataset(raw_data, labels)
    return raw_data, labels

class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SleepStagerChambon2018(nn.Module):
    """Sleep staging architecture from [1]_.

    Convolutional neural network for sleep staging described in [1]_.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    sfreq : float
        EEG sampling frequency.
    n_conv_chs : int
        Number of convolutional channels. Set to 8 in [1]_.
    time_conv_size_s : float
        Size of filters in temporal convolution layers, in seconds. Set to 0.5
        in [1]_ (64 samples at sfreq=128).
    max_pool_size_s : float
        Max pooling size, in seconds. Set to 0.125 in [1]_ (16 samples at
        sfreq=128).
    n_classes : int
        Number of classes.
    input_size_s : float
        Size of the input, in seconds.
    dropout : float
        Dropout rate before the output dense layer.

    References
    ----------
    .. [1] Chambon, S., Galtier, M. N., Arnal, P. J., Wainrib, G., &
           Gramfort, A. (2018). A deep learning architecture for temporal sleep
           stage classification using multivariate and multimodal time series.
           IEEE Transactions on Neural Systems and Rehabilitation Engineering,
           26(4), 758-769.
    """

    def __init__(self, n_channels, sfreq, n_conv_chs=32, time_conv_size_s=0.5,
                 max_pool_size_s=0.125, n_classes=2, input_size_s=30,
                 dropout=0.5):
        super().__init__()

        time_conv_size = int(time_conv_size_s * sfreq)
        # max_pool_size = int(max_pool_size_s * sfreq)
        max_pool_size = 2
        input_size = int(input_size_s * sfreq)
        pad_size = time_conv_size // 2
        self.n_channels = n_channels
        len_last_layer = self._len_last_layer(
            n_channels, input_size, max_pool_size, n_conv_chs)

        self.BN = nn.BatchNorm1d(1)

        if n_channels > 1:
            self.spatial_conv = nn.Conv2d(1, n_channels, (n_channels, 1))

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                1, int(n_conv_chs), (1, time_conv_size), padding=(0, pad_size)),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
            nn.Dropout(dropout),
            nn.Conv2d(
                int(n_conv_chs), int(n_conv_chs), (1, time_conv_size),
                padding=(0, pad_size)),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
            nn.Dropout(dropout)
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(int(len_last_layer), n_classes)
        )


    @staticmethod
    def _len_last_layer(n_channels, input_size, max_pool_size, n_conv_chs):
        return n_channels * (input_size // (max_pool_size ** 2)) * n_conv_chs

    def forward(self, x):
        """Forward pass.

        Parameters
        ---------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """
        if len(x.shape) < 3:
            x = torch.unsqueeze(x, dim=0)

        if self.n_channels > 1:
            x = self.BN(torch.transpose(torch.unsqueeze(x, dim=2), 2, 1))
            x = self.spatial_conv(x)
            x = x.transpose(1, 2)

        x = self.feature_extractor(x)
        return self.fc(x.flatten(start_dim=1))


class EEGClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        # define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # apply layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        return x

def classify_eeg_dft(dft_array_dataset,n_channels,sfreq):
    # Use the GPU if it is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Split the dataset into train and test sets
    train_dataset, test_dataset = random_split(dft_array_dataset, [0.8, 0.2])

    batch_size = 10
    # Create DataLoaders to feed the data to the model in batches
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Define the model
    model = SleepStagerChambon2018(n_channels, sfreq).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    # Train the model
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_dataloader:
            model.train()
            # Perform a forward pass on the batch of data
            predictions = model(x_batch)
            # Compute the loss
            loss = loss_fn(predictions, y_batch)
            # Perform a backward pass to compute the gradients
            loss.backward()
            # Update the model parameters
            optimizer.step()

    # Evaluate the model on the test data
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in test_dataloader:
            # Perform a forward pass on the batch of data
            predictions = model(x_batch)

    return model

def loss_fn(outputs, labels):
    return torch.nn.functional.binary_cross_entropy_with_logits(outputs, labels)

def process_and_plot_eeg_files(directory, seizures):
    # Use glob to find all the .edf files in the directory and its subdirectories
    file_pattern = f'{directory}/*.edf'
    filenames = glob.glob(file_pattern, recursive=True)

    # Load each .edf file and process it using the eeg_dft_array and classify_eeg_dft functions
    for filename in filenames:
        eeg_recording = mne.io.read_raw_edf(filename)
        dft_array_dataset = eeg_dft_array(eeg_recording,seizures)
        labels = classify_eeg_dft(dft_array_dataset, eeg_recording.info['nchan'], eeg_recording.info['sfreq'])


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
            # The string to convert
            str_mm_ss_mm = times_match.group(1)

            # Split the string into minutes, seconds, and milliseconds
            minutes, seconds, milliseconds = str_mm_ss_mm.split(".")

            # Convert the minutes, seconds, and milliseconds to integers
            minutes = int(minutes)
            seconds = int(seconds)
            milliseconds = int(milliseconds)

            # Calculate the value in seconds as a float
            seconds_float = minutes * 60 + seconds + milliseconds / 100
            # Parse a line with the registration and seizure times
            seizures[seizure_number][line.split(':')[0]] = seconds_float

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
            channels, seizures = get_labels(root)
            # Process and plot the EEG recordings in the data directory
            process_and_plot_eeg_files(root, seizures)
