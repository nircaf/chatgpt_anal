import glob
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.fftpack import dct
import mne
import re
from torch.utils.data import Dataset, DataLoader,random_split
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import datetime
import os
import seaborn as sns
import sklearn
import pandas as pd

def eeg_dft_generator(eeg_recording, window_size=1.0):
    # Convert the window size from seconds to the number of samples in the recording
    window_size_samples = int(window_size * eeg_recording.sampling_rate)

    # Iterate over the recording, yielding a window of samples and their DFT at each iteration
    for start in range(0, len(eeg_recording), window_size_samples):
        window = eeg_recording[start:start + window_size_samples]
        dft = np.fft.fft(window)
        yield dft

def seizures_start_time_arr(seizures,dict_key='Seizure start time'):
    arr = []
    for i in seizures:
        arr.append(seizures[i][dict_key])
    return np.array(arr)

def seizures_start_time(seizures,filename,dict_key='Seizure start time'):
    df = pd.DataFrame()
    for i in seizures:
        # get the name from filename path for linux and windows with os
        filenam = os.path.basename(filename)
        # get from seizures only filename dict
        if filenam in seizures[i].values():
            for key in seizures[i]:
                if dict_key.lower() in key.lower():
                    df = pd.concat([df,pd.Series(seizures[i][key])])
    # remove duplicates
    return df.drop_duplicates().to_numpy()

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

def eeg_dft_array(eeg_recording,seizures,filename, window_size=1.0):
    # Convert the window size from seconds to the number of samples in the recording
    window_size_samples = int(window_size * eeg_recording.info['sfreq'])
    batch_size = eeg_recording._raw_lengths[0] // window_size_samples
    # Initialize the array that will store the DFT of the recording
    num_channels = eeg_recording.info['nchan']
    dft_array = np.zeros((batch_size, num_channels,window_size_samples), dtype=np.float64)
    # get raw data index to channel names
    raw_data = pd.DataFrame(eeg_recording.get_data(),index = eeg_recording.ch_names)
    # seizure start and end time array
    seizures_start_time_arr = seizures_start_time(seizures,filename)*eeg_recording.info['sfreq']
    seizures_end_time_arr = seizures_start_time(seizures,filename,dict_key='Seizure end time')*eeg_recording.info['sfreq']
    # Registration start and end time array
    # registration_start_time_arr = seizures_start_time(seizures,dict_key='Registration start time')*eeg_recording.info['sfreq']
    # registration_end_time_arr = seizures_start_time(seizures,dict_key='Registration end time')*eeg_recording.info['sfreq']
    # get raw_data of registration time
    seizures_data = pd.DataFrame(index = eeg_recording.ch_names)
    for i in range(len(seizures_start_time_arr)):
        # get raw_data time equal to half of the seizure time from before and half of the seizure time from after
        start_index = max([0,int(seizures_start_time_arr[i][0] - (seizures_end_time_arr[i][0] - seizures_start_time_arr[i][0])/2)])
        end_index = min([len(raw_data.columns),int(seizures_end_time_arr[i][0] + (seizures_end_time_arr[i][0] - seizures_start_time_arr[i][0])/2)])
        seizures_data = pd.concat([seizures_data,raw_data.iloc[:,start_index:end_index]],ignore_index=True,axis=1)
    labels = np.zeros(seizures_data.shape[1], dtype=np.int64)
    # set labels to 1 for seizure time
    for i in range(len(seizures_start_time_arr)):
        start_index = max([0,int(seizures_start_time_arr[i][0] - (seizures_end_time_arr[i][0] - seizures_start_time_arr[i][0])/2)])
        labels[int(seizures_start_time_arr[i][0]-start_index):int(seizures_end_time_arr[i][0]-start_index)] = 1
    # print percentage of seizure time
    # print(sum(labels)/seizures_data.shape[1]) if seizures_data.shape[1] != 0 else print(0)
    # unsqueeze the dft_array to add a channel dimension
    return seizures_data, labels

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

class BinaryClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def classify_eeg_dft(dft_array_dataset,n_channels,sfreq):
    # Use the GPU if it is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_channels = dft_array_dataset.data.shape[1]
    # Split the dataset into train and test sets
    train_dataset, test_dataset = random_split(dft_array_dataset, [0.8, 0.2])

    batch_size = n_channels
    # Create DataLoaders to feed the data to the model in batches
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Define the model
    model = BinaryClassifier(n_channels, n_channels*2,1).to(device)

def dataframe_to_tensor(dataframe):
    # Convert the dataframe to a numpy array
    data = dataframe.values
    # Convert the numpy array to a FloatTensor torch tensor
    data = torch.from_numpy(data).float()
    return data

def run_torch_model(model,x_data,y_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    batch_size = 128
    # dataframes to tensors
    x_data = dataframe_to_tensor(x_data).to(device)
    y_data = dataframe_to_tensor(y_data).to(device)
    # reshape x_data to be (B,C,H,W)
    x_data_reshape = torch.reshape(x_data,(x_data.shape[0],x_data.shape[1],1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # sklearn train test split
    data_train, data_test, target_train, target_test= sklearn.model_selection.train_test_split(x_data_reshape, y_data, test_size=0.01, random_state=1)
    data_train, data_val, target_train, target_val = sklearn.model_selection.train_test_split(data_train, target_train, test_size=0.25, random_state=1)
    train_dataset = EEGDataset(data_train,target_train)
    val_dataset = EEGDataset(data_val,target_val)
    test_dataset = EEGDataset(data_test,target_test)
    Trainloader = DataLoader(train_dataset,batch_size)
    Valloader = DataLoader(val_dataset,batch_size)
    Testloader = DataLoader(test_dataset,batch_size)
    num_epochs = 100
    # Train the model
    for epoch in range(num_epochs):
        for x_batch, y_batch in Trainloader:
            model.train()
            # Perform a forward pass on the batch of data
            predictions = model(x_batch)
            # Compute the loss
            loss = loss_fn(predictions, y_batch)
            # Perform a backward pass to compute the gradients
            loss.backward()
            # Update the model parameters
            optimizer.step()
            # print loss
        print("Epoch {} - Loss: {:.4f}".format(epoch, loss.detach().item()))

    # Evaluate the model on the test data
    model.eval()
    results = []
    with torch.no_grad():
        for j, data in enumerate(Valloader, 0):
            x_test, y_test = data
            answer = model(x_test.cuda())
            probs=np.exp(answer.cpu().detach().numpy())
            # Calculate the accuracy of the model
            preds = probs.argmax(axis = -1)
            acc=accuracy_score(y_test.cpu().detach().numpy(), preds)
            results.append(acc)
    print('mean accuracy:',np.mean(results))
    # torch.save(model, r'saved_models/{}model.pt'.format(datetime.datetime.now().strftime("%Y_%m_%d")))
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
        seizures_data, labels = eeg_dft_array(eeg_recording,seizures)
        dft_array_dataset = prep_seizures_data_labels(seizures_data, labels)
        # labels = classify_eeg_dft_fft(seizures_data, eeg_recording.info['nchan'], eeg_recording.info['sfreq'])
        labels = classify_eeg_dft(dft_array_dataset, eeg_recording.info['nchan'], eeg_recording.info['sfreq'])

def prep_seizures_data_labels(seizures_data, labels):
    # Use the GPU if it is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seizures_data = torch.tensor(seizures_data.T.values).to(device).float()
    labels = torch.tensor(labels.values).to(device).float()
    dft_array_dataset = EEGDataset(seizures_data, labels)
    return dft_array_dataset

def classify_eeg_dft_fft(seizures_data, n_channels, s_freq):
    # Compute the DFT of the seizures data
    dft_array = torch.fft.fft(seizures_data)

    # Select the frequencies of interest (up to the Nyquist frequency)
    nyquist_freq = s_freq / 2
    freq_idx = dft_array.abs().max(dim=1)[0] >= nyquist_freq

    # Compute the average power spectrum across channels
    avg_power_spectrum = dft_array[:, freq_idx].mean(dim=1)

    # Classify the labels based on the average power spectrum
    labels = (avg_power_spectrum >= 0.5).float()

    return labels

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
    channel_pattern = re.compile(r'Channel (\d+): (.*).*\n')
    seizure_pattern = re.compile(r'Seizure n.*(\d+)')
    filename_pattern = re.compile(r'File \Same: (.*).*\n')
    times_pattern = re.compile(r'.*:.*(\d{2}.\d{2}.\d{2})')
    times_pattern2 = re.compile(r'.*:.*(\d{4}).*seconds')
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
            continue

        seizure_match = seizure_pattern.match(line)
        if seizure_match:
            # Parse a line with seizure information
            seizure_number = int(seizure_match.group(1))
            seizures[seizure_number] = {}


        filename_match = filename_pattern.match(line)
        if filename_match:
            # Parse a line with the filename
            filenam = filename_match.group(1)
            # find number after _ in filename using re
            seizure_number = int(re.findall(r'[-_](\d+)', filenam)[0])
            # seizure_number = int(filenam.split('_')[-1].split('.')[0])
            seizures[seizure_number] = {}
            seizures[seizure_number]['filename'] = filenam

        times_match = times_pattern.match(line)
        if times_match:
            # The string to convert
            str_mm_ss_mm = times_match.group(1)
            # replace : to .
            str_mm_ss_mm = str_mm_ss_mm.replace(':', '.')
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
        else:
            times_match2 = times_pattern2.match(line)
            if times_match2:
                # Parse a line with the registration and seizure times
                seizures[seizure_number][line.split(':')[0]] = int(times_match2.group(1))
    return channels, seizures


def anomaly_detection(data):
    # Convert the data to a tensor
    data = torch.tensor(data).to(torch.float32)

    # Define the dimensions of the input and hidden layers
    input_dim = data.shape[1]
    hidden_dim = int(input_dim / 2)

    # Define the autoencoder model
    autoencoder = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, input_dim),
        torch.nn.Sigmoid()
    )

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters())

    # Train the autoencoder
    for epoch in range(20):
        # Forward pass
        output = autoencoder(data)
        loss = criterion(output, data)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch {} - Loss: {:.4f}".format(epoch, loss.detach().item()))


    # Calculate the reconstruction error for each sample
    error = torch.mean((output - data) ** 2, dim=1)
    print(f'reconstruction error: {torch.mean(error).item()}')
    # Identify the samples with the highest reconstruction error as anomalies
    anomalies = torch.where(error > error.mean() + 2 * error.std())
    torch.save(autoencoder, r'saved_models/{}model.pt'.format(datetime.datetime.now().strftime("%Y_%m_%d")))
    # plot anomalies

    # Take the mean of the data along the columns
    mean_data = data.mean(axis=1)
    # arary of zeros with the same shape as mean_data
    anomal_indices = np.zeros(mean_data.shape)
    # put one where anomalies[0].tolist()
    anomal_indices[anomalies[0].tolist()] = 1
    # Create a scatter plot of the data and color anomal_indices differently
    # plt.scatter(range(data.shape[0]), mean_data, c=anomal_indices)

    # Create a scatter plot of the data and color anomal_indices differently
    sns.scatterplot(x=range(data.shape[0]),y= mean_data, hue=anomal_indices)
    # Set the path to the "Nir_figures" folder
    path = os.path.join("Nir_figures")
    # Create the "Nir_figures" folder if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)
    # Save the figure in the "Nir_figures" folder
    plt.savefig(os.path.join(path, "figure.png"))
    # Return the anomalies as a list of indices
    return anomalies[0].tolist()

if __name__ == '__main__':
    # Use the GPU if it's available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
