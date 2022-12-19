import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import neo
import os
from tkinter import Tk
from tkinter import filedialog
import seaborn as sns; sns.set_theme()
from scipy.spatial.distance import cdist

import scipy.io
nsx = 3
if nsx == 3:
    fs = 500
elif nsx == 6:
    fs = 30e3
num_of_sec_total = 5
seg_dur_in_sec = 0.5


cwd = os.curdir
Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
workdir = filedialog.askdirectory(
                title=f'Please choose the directory which contains the files from Eichilove',
                initialdir=os.path.abspath(os.path.join(cwd, os.pardir,
                                                        os.pardir)))  # show an "Open" dialog box and return the path to the selected fileonlyfiles = [os.path.join(cwd, f) for f in os.listdir(cwd) if
onlyfiles = np.unique([os.path.join(workdir, f.split('.')[0]) for f in os.listdir(workdir) if os.path.isfile(os.path.join(workdir, f))])

data_list = list()
flag = 0
for inx_file, file in enumerate(onlyfiles):
    try:
        reader = neo.BlackrockIO(filename=file, nsx_to_load=nsx)
        print(file)
        cur_data = reader.nsx_datas.get(nsx).get(0)
        nev_data = reader.nev_data.get('NonNeural')[0]
        for cur_nev_data in nev_data:
            evnt_time_stamp = int(cur_nev_data[0] * (fs / 30e3))
            if evnt_time_stamp <= num_of_sec_total * fs:
                continue
            near_ev_data = cur_data[max(int(evnt_time_stamp - num_of_sec_total * fs), 0):(evnt_time_stamp), :]
            data_list.append(np.reshape(near_ev_data, (int(num_of_sec_total / seg_dur_in_sec), int(seg_dur_in_sec * fs), cur_data.shape[1]), order='F'))

        if flag == 0:
            ch_names = reader.raw_annotations['blocks'][0]['segments'][0]['signals'][0]['__array_annotations__'][
                'channel_names']
            flag = 1
        del reader
    except:
        pass

chl_names = []
for ch in ch_names:
    if len(ch.split('-')) == 1:
        chl_names.append(ch)
    else:
        chl_names.append(ch.split('-')[-1])

data_mat = np.concatenate(data_list)
mu = np.mean(data_mat, axis=0)
std = np.std(data_mat, axis=0)

cov = np.cov(mu)
norm_data_mat = (data_mat - np.expand_dims(mu, axis=0)) / (np.expand_dims(std, axis=0) + 1e-10)
results = cdist(norm_data_mat[6565, :, :].T, norm_data_mat[6565, :, :].T, 'mahalanobis', VI=cov)
ax_heatmap = plt.figure()
ax_heatmap = sns.heatmap(results, xticklabels=chl_names, yticklabels=chl_names)
sum = np.sum(results, axis=0)
df = pd.DataFrame({'Channels': chl_names,
                   'Mahalanobis': sum})
ax1 = plt.figure()
ax1 = sns.lineplot(data=df, x="Channels", y="Mahalanobis")
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
#create some fake data

# for ch_inx, ch in enumerate(cur_data.T):
#     if ch_inx == 0:
#         ax_ = plt.figure()
#     ax_ = plt.plot(np.arange(len(ch)), ch)
