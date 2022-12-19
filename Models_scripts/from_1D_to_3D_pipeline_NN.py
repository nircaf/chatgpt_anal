import torch
import os

import torch.optim as optim
import torchvision

import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
import numpy as np
import copy
import torch.nn.functional as F
import pickle
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import matplotlib.pyplot as plt

from torchesn.nn import ESN
from torchesn import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
# from EchoTorch_dev.echotorch.modules import esn
import numpy as np
# from sklearn.manifold import TSNE
from tsne_torch import TorchTSNE as TSNE
from SSVEP_Neural_Generative_Models.src.EEG_VAE import *
# from AutoEncoder import Encoder_single_ch as Encoder

from Models_scripts.MemoryBank_UnSperVisend_Classification import *
from Models_scripts.Soft_NN_loss import *
from Models_scripts.MS_MDA_utils import *

def NeuroBraveLoss(x=None, labels=None, mode='Mahalanobis'):
    b, n = x.size()
    A = torch.zeros((b, b))
    if 'Mahalanobis' in mode:
        for ii in range(b):
            for jj in range(ii, b):
                delta = torch.unsqueeze(F.sigmoid(x[ii, :]) - F.sigmoid(x[jj, :]) + 1/n, dim=1)
                if labels[ii] == labels[jj]:
                    A[ii, jj] = torch.mm(delta.t(), delta) / n
                else:
                    A[ii, jj] = 1 - torch.mm(delta.t(), delta) / n

        res = -torch.log((1 / ((b + 1) * b / 2)) * torch.sum(A))
    elif 'Soft-Nearest Neighbors Loss' in mode:
        res = 0
        t = 0.07
        for ii in range(b):
            cur_lbl = labels[ii]


    return res

def normalized_batch(batch_x=None, std=None, mu=None):
    assert ((batch_x.shape[1] == len(std)) or (batch_x.shape[1] == len(mu))), \
        'Number of channels must be equals to the Mu and Sigma normalization factors'
    transform = torchvision.transforms.Compose([
                torchvision.transforms.Normalize(mu, std + 1e-10)
            ])
    batch_x = torch.unsqueeze(batch_x, dim=-1)
    res_batch = transform(batch_x)
    return torch.squeeze(res_batch)

def normalized_batch_mu_only(batch_x=None, std=None, mu=None):
    assert ((batch_x.shape[1] == len(std)) or (batch_x.shape[1] == len(mu))), \
        'Number of channels must be equals to the Mu and Sigma normalization factors'
    transform = torchvision.transforms.Compose([
                torchvision.transforms.Normalize(mu, torch.ones_like(mu))
            ])
    batch_x = torch.unsqueeze(batch_x, dim=-1)
    res_batch = transform(batch_x)
    return torch.squeeze(res_batch)

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def my_loss(output, target):
    loss = torch.mean((torch.floor(output) - target)**2)
    return loss

def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def evaluate_eeg(model, X, Y, params=["acc"]):
    results = []
    batch_size = 16

    Y = Y.cpu().numpy()

    predicted = []
    if torch.cuda.is_available():
        cuda0 = torch.device('cuda:0')
        X = X.to(device=cuda0, dtype=torch.float32)

    for i in range(int(len(X) / batch_size)):
        s = i * batch_size
        e = i * batch_size + batch_size

        inputs = X[s:e]
        pred = model(inputs)

        predicted.append(pred.data.cpu().numpy())

    inputs = X
    predicted = model(inputs)

    predicted, predicted_fixed = torch.max(torch.softmax(predicted, dim=1), dim=1).values.cpu().detach().numpy(), torch.max(torch.softmax(predicted, dim=1), dim=1).indices.cpu().detach().numpy()
     # = torch.argmax(predicted, dim=1).data.cpu().numpy()

    for param in params:
        if param == 'acc':
            results.append(accuracy_score(Y, predicted_fixed))
        if param == "auc":
            results.append(roc_auc_score(Y, predicted))
        if param == "recall":
            results.append(recall_score(Y, predicted_fixed))
        if param == "precision":
            results.append(precision_score(Y, predicted_fixed))
        if param == "fmeasure":
            precision = precision_score(Y, predicted_fixed)
            recall = recall_score(Y, predicted_fixed)
            results.append(2 * precision * recall / (precision + recall))
    return results
# Authors: Robin Tibor Schirrmeister <robintibor@gmail.com>
#          Tonio Ball
#
# License: BSD-3

import numpy as np

import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import elu
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Weight initilaztion
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)

def weights_init_normal(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.Conv1d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def weights_init_uniform(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight.data)
    elif type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform(m.weight.data)
    elif type(m) == nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class SleepStagerChambon2018_CFE(nn.Module):
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
                 max_pool_size_s=0.125, n_classes=5, input_size_s=30,
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

        self.BN = nn.BatchNorm2d(1)

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
            nn.Linear(int(len_last_layer), min(int(len_last_layer//8), 256))
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
class DSFE(nn.Module):
    def __init__(self, n_channels=4, sfreq=250, n_conv_chs=32, time_conv_size_s=0.5,
                 max_pool_size_s=0.125, n_classes=5, input_size_s=30,
                 dropout=0.5):
        super(DSFE, self).__init__()
        max_pool_size = 2
        input_size = int(input_size_s * sfreq)
        len_last_layer = self._len_last_layer(
            n_channels, input_size, max_pool_size, n_conv_chs)
        self.module = nn.Sequential(
            nn.Linear(min(int(len_last_layer//8), 256), 32),
            # nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
    def _len_last_layer(self, n_channels, input_size, max_pool_size, n_conv_chs):
        return n_channels * (input_size // (max_pool_size ** 2)) * n_conv_chs
    def forward(self, x):
        x = self.module(x)
        return x

class MSMDAERNet(nn.Module):
    def __init__(self, n_channels=0, sfreq=0, pretrained=False, number_of_source=15, number_of_category=2, n_conv_chs=32, time_conv_size_s=0.5,
                 max_pool_size_s=0.125, n_classes=5, input_size_s=30,
                 dropout=0.5):
        super(MSMDAERNet, self).__init__()
        self.sharedNet = SleepStagerChambon2018_CFE(n_channels=n_channels, sfreq=sfreq, n_conv_chs=n_conv_chs, time_conv_size_s=time_conv_size_s,
                 max_pool_size_s=max_pool_size_s, n_classes=n_classes, input_size_s=input_size_s,
                 dropout=dropout)
        # for i in range(1, number_of_source):
        #     exec('self.DSFE' + str(i) + '=DSFE()')
        #     exec('self.cls_fc_DSC' + str(i) + '=nn.Linear(32,' + str(number_of_category) + ')')
        for i in range(number_of_source):
            exec('self.DSFE' + str(i) + '=DSFE()')
            exec('self.cls_fc_DSC' + str(i) + '=nn.Linear(32,' + str(number_of_category) + ')')

    def forward(self, data_src, number_of_source, data_tgt=0, label_src=0, mark=0):
        '''
        description: take one source data and the target data in every forward operation.
            the mmd loss is calculated between the source data and the target data (both after the DSFE)
            the discrepency loss is calculated between all the classifiers' results (test on the target data)
            the cls loss is calculated between the ground truth label and the prediction of the mark-th classifier
            之所以target data每一条线都要过一遍是因为要计算discrepency loss, mmd和cls都只要mark-th那条线就行
        param {type}:
            mark: int, the order of the current source
            data_src: take one source data each time
            number_of_source: int
            label_Src: corresponding label
            data_tgt: target data
        return {type}
        '''
        mmd_loss = 0
        disc_loss = 0
        data_tgt_DSFE = []
        if self.training == True:
            # common feature extractor
            data_src_CFE = self.sharedNet(data_src)
            data_tgt_CFE = self.sharedNet(data_tgt)
            # Each domian specific feature extractor
            # to extract the domain specific feature of target data
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                data_tgt_DSFE_i = eval(DSFE_name)(data_tgt_CFE)
                data_tgt_DSFE.append(data_tgt_DSFE_i)
            # Use the specific feature extractor
            # to extract the source data, and calculate the mmd loss
            DSFE_name = 'self.DSFE' + str(mark)
            data_src_DSFE = eval(DSFE_name)(data_src_CFE)
            # mmd_loss += utils.mmd(data_src_DSFE, data_tgt_DSFE[mark])
            mmd_loss += utils.mmd_linear(data_src_DSFE, data_tgt_DSFE[mark])
            # discrepency loss
            for i in range(len(data_tgt_DSFE)):
                if i != mark:
                    disc_loss += torch.mean(torch.abs(
                        F.softmax(data_tgt_DSFE[mark], dim=1) -
                        F.softmax(data_tgt_DSFE[i], dim=1)
                    ))
            # domain specific classifier and cls_loss
            DSC_name = 'self.cls_fc_DSC' + str(mark)
            pred_src = eval(DSC_name)(data_src_DSFE)
            cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src.squeeze())

            return cls_loss, mmd_loss, disc_loss

        else:
            data_CFE = self.sharedNet(data_src)
            pred = []
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                DSC_name = 'self.cls_fc_DSC' + str(i)
                feature_DSFE_i = eval(DSFE_name)(data_CFE)
                pred.append(eval(DSC_name)(feature_DSFE_i))

            return pred


class MSMDAER():
    def __init__(self, model=0, source_loaders=0, target_train_loader=0, target_loader=0, batch_size=64, iteration=10000, lr=0.001, momentum=0.9, log_interval=10, Best_model_save_path=0, criterion=0):
        self.Best_model_save_path = Best_model_save_path
        self.model = model
        self.model.to(device)
        self.source_loaders = source_loaders
        self.target_train_loader = target_train_loader
        self.target_loader = target_loader
        self.batch_size = batch_size
        self.iteration = iteration
        self.lr = lr
        self.momentum = momentum
        self.log_interval = log_interval
        self.criterion = criterion

    def __getModel__(self):
        return self.model

    def train(self):
        # best_model_wts = copy.deepcopy(model.state_dict())
        source_iters = []
        for i in range(len(self.source_loaders)):
            source_iters.append(iter(self.source_loaders[i]))
        target_iter = iter(self.target_train_loader)
        correct = 0
        torch.manual_seed(1221)
        self.model.apply(weights_init_uniform)
        for i in range(1, self.iteration+1):
            self.model.train()
            # LEARNING_RATE = self.lr / math.pow((1 + 10 * (i - 1) / (self.iteration)), 0.75)
            LEARNING_RATE = self.lr
            # if (i - 1) % 100 == 0:
                # print("Learning rate: ", LEARNING_RATE)
            # optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=self.momentum)
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=LEARNING_RATE)

            for j in range(len(source_iters)):
                try:
                    source_data, source_label = next(source_iters[j])
                except Exception as err:
                    source_iters[j] = iter(self.source_loaders[j])
                    source_data, source_label = next(source_iters[j])
                try:
                    target_data, _ = next(target_iter)
                except Exception as err:
                    target_iter = iter(self.target_train_loader)
                    target_data, _ = next(target_iter)
                source_data, source_label = source_data.to(device=device, dtype=torch.float32), source_label.to(device=device, dtype=torch.int64)
                target_data = target_data.to(device=device, dtype=torch.float32)

                optimizer.zero_grad()
                cls_loss, mmd_loss, l1_loss = self.model(source_data, number_of_source=len(
                    source_iters), data_tgt=target_data, label_src=source_label, mark=j)
                gamma = 2 / (1 + math.exp(-10 * (i) / (self.iteration))) - 1
                beta = gamma/100
                # loss = cls_loss + gamma * (mmd_loss + l1_loss)
                loss = cls_loss + gamma * mmd_loss + beta * l1_loss
                # loss = cls_loss + gamma * (mmd_loss)
                # writer.add_scalar('Loss/training cls loss', cls_loss, i)
                # writer.add_scalar('Loss/training mmd loss', mmd_loss, i)
                # writer.add_scalar('Loss/training l1 loss', l1_loss, i)
                # writer.add_scalar('Loss/training gamma', gamma, i)
                # writer.add_scalar('Loss/training loss', loss, i)
                loss.backward()
                optimizer.step()

                if i % self.log_interval == 0:
                    print('Train source' + str(j) + ', iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_loss: {:.6f}\tmmd_loss {:.6f}\tl1_loss: {:.6f}'.format(
                        i, 100.*i/self.iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()
                    )
                    )
                # if i % log_interval == 0:
                #     print('Train source' + str(j) + ', iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_loss: {:.6f}\tmmd_loss {:.6f}\tl1_loss: {:.6f}'.format(
                #         i, 100.*i/self.iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()
                #         )
                #     )
            if i % (self.log_interval * 20) == 0:
                t_correct, test_loss, correct_num = self.test(i)
                if t_correct > correct:
                    print(f'best val ACC {correct:.4f} -> {t_correct:.4f}')
                    correct = t_correct
                    # Save the best model
                    if self.Best_model_save_path is None:
                        best_model = copy.deepcopy(self.model)
                    else:
                        torch.save({
                            'epoch': i,
                            'model_state_dict': self.model.to('cpu').state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'valid_loss': test_loss,
                            'train_loss': loss.item(),
                        }, self.Best_model_save_path)
                        try:
                            with open(self.Best_model_save_path_pickle, 'wb') as f:
                                pickle.dump(pickle.dumps(self.model), f)
                        except:
                            pass
                        self.model = self.model.to(device)
                    waiting = 0
                else:
                    waiting += 1


                # print('to target max correct: ', correct.item(), "\n")
        return 100. * correct_num / len(self.target_loader.dataset)

    def test(self, i):
        self.model.eval()
        test_loss = 0
        correct = 0
        corrects = []
        for i in range(len(self.source_loaders)):
            corrects.append(0)
        with torch.no_grad():
            for data, target in self.target_loader:
                data = data.to(device=device, dtype=torch.float32)
                target = target.to(device=device, dtype=torch.int64)
                preds = self.model(data, len(self.source_loaders))
                for i in range(len(preds)):
                    preds[i] = F.softmax(preds[i], dim=1)
                pred = sum(preds)/len(preds)
                test_loss += F.nll_loss(F.log_softmax(pred,
                                        dim=1), target.squeeze()).item()
                pred = pred.data.max(1)[1]
                correct += pred.eq(target.data.squeeze()).cpu().sum()
                for j in range(len(self.source_loaders)):
                    pred = preds[j].data.max(1)[1]
                    corrects[j] += pred.eq(target.data.squeeze()).cpu().sum()

            test_loss /= len(self.target_loader.dataset)
            Correct = correct / len(self.target_loader.dataset)
            # writer.add_scalar("Test/Test loss", test_loss, i)

            # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            #     test_loss, correct, len(self.target_loader.dataset),
            #     100. * correct / len(self.target_loader.dataset)
            # ))
            # for n in range(len(corrects)):
            #     print('Source' + str(n) + 'accnum {}'.format(corrects[n]))
        return Correct, test_loss, correct


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hn=None, cn=None):
        x = torch.transpose(x, dim0=1, dim1=2)
        if hn is None:
            # Initializing hidden state for first input with zeros
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

            # Initializing cell state for first input with zeros
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

            if torch.cuda.is_available():
                h0 = h0.cuda()
                c0 = c0.cuda()
            # We need to detach as we are doing truncated backpropagation through time (BPTT)
            # If we don't, we'll backprop all the way to the start even after going through another batch
            # Forward propagation by passing in the input, hidden state, and cell state into the model
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        else:
            try:
                out, (hn, cn) = self.lstm(x, (hn.detach(), cn.detach()))
            except:
                # Initializing hidden state for first input with zeros
                h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

                # Initializing cell state for first input with zeros
                c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

                if torch.cuda.is_available():
                    h0 = h0.cuda()
                    c0 = c0.cuda()

                out, (_, _) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        try:
            return out, hn, cn
        except:
            return out

class EEGNet(nn.Module):
    def __init__(self, n_classes=3, num_ch=64, sig_time_stamps=250, args=None):
        super(EEGNet, self).__init__()
        self.n_classes = n_classes
        self.args = args
        self.T = sig_time_stamps

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, num_ch), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        self.fc1 = nn.Linear(4 * 2 * 12, n_classes)

    def forward(self, x):
        # Layer 1
        try:
            x = F.elu(self.conv1(x))
        except:
            x = F.elu(self.conv1(torch.unsqueeze(x, dim=1).permute(0, 1, 3, 2)))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        # FC Layer
        x = x.reshape(-1, 4 * 2 * 12)
        x = self.fc1(x)
        return x




class EEGNet_evaluation():
    def __init__(self, args=None):
        super(EEGNet_evaluation, self).__init__()
        self.args = args
        self.batch_size = self.args.batch_size
        self.batch_size_v = self.args.batch_size_val
        self.n_classes = self.args.n_classes

        if self.args.use_cuda and torch.cuda.is_available():
            self.model = EEGNet(n_classes=self.n_classes, sig_time_stamps=self.args.raw_time_duration, args=self.args).cuda()
        else:
            self.model = EEGNet(n_classes=self.n_classes, sig_time_stamps=self.args.raw_time_duration, args=self.args)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def Run_eegnet(self, X_train, y_train, X_val, y_val, X_test, y_test, optimizer, criterion):
        for epoch in range(self.args.n_epoch):  # loop over the dataset multiple times
            print("\nEpoch ", epoch)
            self.model.train()
            running_loss = 0.0
            print_freq = 100
            for i in range(int(len(X_train) / self.batch_size - 1)):
                s = i * self.batch_size
                e = i * self.batch_size + self.batch_size

                inputs = torch.from_numpy(X_train[s:e])
                labels = torch.FloatTensor(np.array([y_train[s:e]]).T * 1.0)

                # wrap them in Variable
                inputs, labels = Variable(inputs.to(self.device)), Variable(labels.to(self.device))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(torch.unsqueeze(torch.transpose(inputs, dim0=1, dim1=2), axis=1).float())
                loss = criterion(outputs, labels[:, 0].long())
                loss.backward()

                optimizer.step()

                running_loss += loss.item()

                if i % print_freq == 0:
                    print(f'[batch / total DS]: [{i}/{int(len(X_train) / self.batch_size - 1)}], Avg Loss: {running_loss / (i+1)}')

            # Validation accuracy
            print('\n')
            params = ["acc", "fmeasure"]
            print(params)
            print("Training Loss ", (running_loss / int(len(X_train) / self.batch_size - 1)))
            print("Train - ", self.eegnet_evaluate(X_train, y_train, params, mode='train'))
            print("Validation - ", self.eegnet_evaluate(X_val, y_val, params, mode='eval'))
            print("Test - ", self.eegnet_evaluate(X_test, y_test, params, mode='eval'))

        return self.model

    def eegnet_evaluate(self, X, Y, params=["acc"], mode='train'):
        results = torch.zeros((len(params), 1))
        if mode == 'train:':
            batch_size = self.batch_size
            self.model.train()
        else:
            batch_size = self.batch_size_v
            self.model.eval()

        predicted = []

        for i in range(int(len(X) / batch_size)):
            s = i * batch_size
            e = i * batch_size + batch_size

            inputs = Variable(torch.from_numpy(X[s:e]).cuda(0))
            pred = self.model(torch.unsqueeze(torch.transpose(inputs, dim0=1, dim1=2), axis=1).float())

            predicted.append(torch.argmax(pred.data).cpu().numpy())

        for param_inx, param in enumerate(params):
            if param == 'acc':
                results[param_inx] += accuracy_score(Y, np.asarray(predicted))
            if param == "auc":
                results[param_inx] += roc_auc_score(Y, np.asarray(predicted))
            if param == "recall":
                results[param_inx] += recall_score(Y, np.asarray(predicted))
            if param == "precision":
                results[param_inx] += precision_score(Y, np.asarray(predicted))
            if param == "fmeasure":
                precision = precision_score(Y, np.asarray(predicted), average='micro')
                recall = recall_score(Y, np.asarray(predicted), average='micro')
                results[param_inx] += (2 * precision * recall / (precision + recall))

        # inputs = Variable(torch.from_numpy(X).to(self.device))
        # predicted = self.model(torch.unsqueeze(torch.transpose(inputs, dim0=1, dim1=2), axis=1).float())
        #
        # predicted = predicted.data.cpu().numpy()


        return results

class t_SNE_Net(nn.Module):
    def __init__(self, args=None):
      super(t_SNE_Net, self).__init__()

      # Designed to ensure that adjacent pixels are either all 0s or all active
      # with an input probability
      self.fc_first_input_size = args.raw_time_duration
      self.fc_last_output_size = args.t_SNE_n_components
      self.dropout1 = nn.Dropout(0.25)
      self.dropout2 = nn.Dropout(0.5)

      self.relu = nn.ReLU()

      # First fully connected layer
      self.fc1 = nn.Linear(2*self.fc_first_input_size, 512)
      # Second fully connected layer that outputs our 10 labels
      self.fc2 = nn.Sequential(
          nn.Dropout(0.5),
          nn.Linear(512, 1024)
      )
      self.fc3 = nn.Sequential(
          nn.Dropout(0.5),
          nn.Linear(1024, 512)
      )
      self.fc4 = nn.Sequential(
          nn.Dropout(0.5),
          nn.Linear(512, 2*self.fc_last_output_size)
      )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        out = self.fc4(x)

        return out


class MSMDAERNet(nn.Module):
    def __init__(self, n_channels, sfreq, pretrained=False, number_of_source=15, number_of_category=2, n_conv_chs=32, time_conv_size_s=0.5,
                 max_pool_size_s=0.125, n_classes=5, input_size_s=30,
                 dropout=0.5, criterion=0):
        super(MSMDAERNet, self).__init__()
        self.sharedNet = SleepStagerChambon2018_CFE(n_channels=n_channels, sfreq=sfreq, n_conv_chs=n_conv_chs, time_conv_size_s=time_conv_size_s,
                 max_pool_size_s=max_pool_size_s, n_classes=n_classes, input_size_s=input_size_s,
                 dropout=dropout)
        self.criterion = criterion
        # for i in range(1, number_of_source):
        #     exec('self.DSFE' + str(i) + '=DSFE()')
        #     exec('self.cls_fc_DSC' + str(i) + '=nn.Linear(32,' + str(number_of_category) + ')')
        for i in range(number_of_source):
            exec('self.DSFE' + str(i) + '=DSFE(n_channels=n_channels, sfreq=sfreq, n_conv_chs=n_conv_chs, time_conv_size_s=time_conv_size_s, \
                 max_pool_size_s=max_pool_size_s, n_classes=n_classes, input_size_s=input_size_s, \
                 dropout=dropout)')
            exec('self.cls_fc_DSC' + str(i) + '=nn.Linear(32,' + str(number_of_category) + ')')

    def forward(self, data_src, number_of_source, data_tgt=0, label_src=0, mark=0):
        '''
        description: take one source data and the target data in every forward operation.
            the mmd loss is calculated between the source data and the target data (both after the DSFE)
            the discrepency loss is calculated between all the classifiers' results (test on the target data)
            the cls loss is calculated between the ground truth label and the prediction of the mark-th classifier
            之所以target data每一条线都要过一遍是因为要计算discrepency loss, mmd和cls都只要mark-th那条线就行
        param {type}:
            mark: int, the order of the current source
            data_src: take one source data each time
            number_of_source: int
            label_Src: corresponding label
            data_tgt: target data
        return {type}
        '''
        global last_mmd_loss
        mmd_loss = 0
        disc_loss = 0
        data_tgt_DSFE = []
        if self.training == True:
            # common feature extractor
            data_src_CFE = self.sharedNet(data_src)
            data_tgt_CFE = self.sharedNet(data_tgt)
            # Each domian specific feature extractor
            # to extract the domain specific feature of target data
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                data_tgt_DSFE_i = eval(DSFE_name)(data_tgt_CFE)
                data_tgt_DSFE.append(data_tgt_DSFE_i)
            # Use the specific feature extractor
            # to extract the source data, and calculate the mmd loss
            DSFE_name = 'self.DSFE' + str(mark)
            # print(data_src_CFE.shape)
            try:
                data_src_DSFE = eval(DSFE_name)(data_src_CFE)
            except:
                pass
            # mmd_loss += utils.mmd(data_src_DSFE, data_tgt_DSFE[mark])
            try:
                last_mmd_loss = mmd_linear(data_src_DSFE, data_tgt_DSFE[mark])
                mmd_loss += last_mmd_loss
            except:
                mmd_loss += last_mmd_loss

            # discrepency loss
            for i in range(len(data_tgt_DSFE)):
                if i != mark:
                    disc_loss += torch.mean(torch.abs(
                        F.softmax(data_tgt_DSFE[mark], dim=1) -
                        F.softmax(data_tgt_DSFE[i], dim=1)
                    ))
            # domain specific classifier and cls_loss
            DSC_name = 'self.cls_fc_DSC' + str(mark)
            pred_src = eval(DSC_name)(data_src_DSFE)

            try:
                cls_loss = self.criterion(pred_src, label_src)
            except:
                cls_loss = self.criterion(pred_src, torch.max(label_src, 1)[0])
            # cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src.squeeze())

            return cls_loss, mmd_loss, disc_loss

        else:
            data_CFE = self.sharedNet(data_src)
            pred = []
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                DSC_name = 'self.cls_fc_DSC' + str(i)
                feature_DSFE_i = eval(DSFE_name)(data_CFE)
                pred.append(eval(DSC_name)(feature_DSFE_i))

            return pred

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        if torch.cuda.is_available():
            h0 = h0.cuda()


        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # # so that it can fit into the fully connected layer
        # out = out[:, -1, :]
        #
        # # Convert the final state to our desired output shape (batch_size, output_dim)
        # out = self.fc(out)

        return out

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
                 max_pool_size_s=0.125, n_classes=5, input_size_s=30,
                 dropout=0.5):
        super().__init__()

        time_conv_size = int(time_conv_size_s * sfreq)
        # max_pool_size = int(max_pool_size_s * sfreq)
        max_pool_size = 2
        input_size = int(input_size_s * sfreq)
        pad_size = time_conv_size // 2
        self.n_channels = n_channels
        len_last_layer = n_channels**2
        # len_last_layer = self._len_last_layer(
        #     n_channels, input_size, max_pool_size, n_conv_chs)

        self.BN = nn.BatchNorm2d(1)

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
        return self.fc(x.flatten(start_dim=1)) # n_channels * (input_size // (max_pool_size ** 2)) * n_conv_chs


class SleepStagerChambon2018_domain_adaptation(nn.Module):
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
                 max_pool_size_s=0.125, n_classes=5, input_size_s=30,
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

        self.BN = nn.BatchNorm2d(1)

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

class SleepStagerChambon2018_transfer_learning(nn.Module):
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
                 max_pool_size_s=0.125, n_classes=5, input_size_s=30,
                 dropout=0.5):
        super().__init__()

        time_conv_size = int(time_conv_size_s * sfreq)
        # max_pool_size = int(max_pool_size_s * sfreq)
        max_pool_size = 2
        input_size = int(input_size_s * sfreq)
        pad_size = time_conv_size // 2
        self.n_channels_target = 16
        len_last_layer = self._len_last_layer(
            self.n_channels_target, input_size, max_pool_size, n_conv_chs)

        self.conv_fixed_ch = nn.Conv2d(n_channels, self.n_channels_target, 1)

        self.BN = nn.BatchNorm2d(1)

        if n_channels > 1:
            self.spatial_conv = nn.Conv2d(1, self.n_channels_target, (self.n_channels_target, 1))

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

    def forward(self, x): # x.shape -> [B, C, L]
        """Forward pass.

        Parameters
        ---------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """
        if len(x.shape) < 3:
            x = torch.unsqueeze(x, dim=0)

        x = torch.unsqueeze(x, dim=1) #[B, 1, C, L]
        x = x.transpose(1, 2) # [B, C, 1, L]
        x = self.conv_fixed_ch(x) # [B, C*, 1, L]
        x = x.transpose(1, 2) # [B, 1, C*, L]
        x = self.BN(x) # [B, 1, C*, L]
        x = self.spatial_conv(x)
        x = x.transpose(1, 2)

        x_before_fc = self.feature_extractor(x)
        return self.fc(x_before_fc.flatten(start_dim=1)), x_before_fc.flatten(start_dim=1)

class SleepStagerChambon2018_fusion(nn.Module):
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

    def __init__(self, n_channels_teacher, n_channels_student, sfreq, n_conv_chs=32, time_conv_size_s=0.5,
                 max_pool_size_s=0.125, n_classes=5, input_size_s=30,
                 dropout=0.5):
        super().__init__()

        time_conv_size = int(time_conv_size_s * sfreq)
        # max_pool_size = int(max_pool_size_s * sfreq)
        max_pool_size = 2
        input_size = int(input_size_s * sfreq)
        pad_size = time_conv_size // 2
        self.teacer_2_learning_factor = 1.25
        self.n_channels_teacher = n_channels_teacher
        self.n_channels_student = n_channels_student
        self.n_channels_learning = int(self.teacer_2_learning_factor * n_channels_teacher)
        len_last_layer = self._len_last_layer(
            self.n_channels_learning, input_size, max_pool_size, n_conv_chs)

        self.BN = nn.BatchNorm2d(1)

        if n_channels_teacher > 1:
            self.spatial_conv_teacher = nn.Conv2d(1, self.n_channels_learning, (self.n_channels_teacher, 1))
        if n_channels_student > 1:
            self.spatial_conv_student = nn.Conv2d(1, self.n_channels_learning, (self.n_channels_student, 1))

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

    def forward(self, x, mode='teacher'):
        """Forward pass.

        Parameters
        ---------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """
        if len(x.shape) < 3:
            x = torch.unsqueeze(x, dim=0)

        if 'teacher' in mode:
            n_channels = self.n_channels_teacher
        elif 'student' in mode:
            n_channels = self.n_channels_student

        if n_channels > 1:
            x = self.BN(torch.transpose(torch.unsqueeze(x, dim=2), 2, 1))
            if 'teacher' in mode:
                x = self.spatial_conv_teacher(x)
            elif 'student' in mode:
                x = self.spatial_conv_student(x)
            x = x.transpose(1, 2)

        x = self.feature_extractor(x)
        return self.fc(x.flatten(start_dim=1))

class SleepStagerChambon2018_super_unsuper_vised(nn.Module):
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
                 max_pool_size_s=0.125, n_classes=5, input_size_s=30,
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

        self.BN = nn.BatchNorm2d(1)

        if n_channels > 1:
            self.spatial_conv = nn.Sequential(nn.Conv2d(1, n_channels, (n_channels, 1)),
                                              nn.ReLU())


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
            x = torch.transpose(torch.unsqueeze(x, dim=2), 2, 1)
            x = self.spatial_conv(x)
            x = x.transpose(1, 2)

        x = self.feature_extractor(x)
        return self.fc(x.flatten(start_dim=1)), x.flatten(start_dim=1)

class SleepStagerChambon2018_with_unsupervides(nn.Module):
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
                 max_pool_size_s=0.125, n_classes=5, input_size_s=30,
                 dropout=0.5, args=None):
        super().__init__()

        time_conv_size = int(time_conv_size_s * sfreq)
        # max_pool_size = int(max_pool_size_s * sfreq)
        max_pool_size = 2
        input_size = int(input_size_s * sfreq)
        pad_size = time_conv_size // 2
        self.n_channels = n_channels
        len_last_layer = self._len_last_layer(
            n_channels, input_size, max_pool_size, n_conv_chs)

        self.BN = nn.BatchNorm2d(1)
        self.memory_bank_base = MemoryBank(args.batch_size,
                                int(len_last_layer),
                                n_classes, 0.1)

        if n_channels > 1:
            self.spatial_conv = nn.Conv2d(1, n_channels, (n_channels, 1))

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                1, int(n_conv_chs), (1, time_conv_size), padding=(0, pad_size)),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
            # nn.Dropout(dropout),
            nn.Conv2d(
                int(n_conv_chs), int(n_conv_chs), (1, time_conv_size),
                padding=(0, pad_size)),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
            # nn.Dropout(dropout)
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


class SleepStagerChambon2018_deeper(nn.Module):
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
                 max_pool_size_s=0.125, n_classes=5, input_size_s=30,
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

        self.BN = nn.BatchNorm2d(1)

        if n_channels > 1:
            self.spatial_conv = nn.Conv2d(1, n_channels, (n_channels, 1))

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                1, int(n_conv_chs), (1, time_conv_size), padding=(0, pad_size)),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
            # nn.Dropout(dropout),
            nn.Conv2d(
                int(n_conv_chs), int(4*n_conv_chs), (1, time_conv_size),
                padding=(0, pad_size)),
            nn.ReLU(),
            # nn.Conv2d(
            #     int(8*n_conv_chs), int(16 * n_conv_chs), (1, time_conv_size),
            #     padding=(0, pad_size)),
            # nn.ReLU(),
            # nn.Conv2d(
            #     int(16*n_conv_chs), int(8 * n_conv_chs), (1, time_conv_size),
            #     padding=(0, pad_size)),
            # nn.ReLU(),
            nn.Conv2d(
                int(4*n_conv_chs), int(n_conv_chs), (1, time_conv_size),
                padding=(0, pad_size)),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
            # nn.Dropout(dropout)
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


class SleepStagerChambon2018_with_gru(nn.Module):
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
    0time_conv_size_s : float
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
                 max_pool_size_s=0.125, n_classes=5, input_size_s=30,
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

        self.BN = nn.BatchNorm2d(1)

        if n_channels > 1:
            self.spatial_conv = nn.Conv2d(1, n_channels, (n_channels, 1))

        if torch.cuda.is_available():
            self.gru = GRUModel(input_dim=n_channels, hidden_dim=n_channels, layer_dim=1, output_dim=n_channels, dropout_prob=0.35).cuda()
        else:
            self.gru = GRUModel(input_dim=n_channels, hidden_dim=n_channels, layer_dim=1, output_dim=n_channels,
                                dropout_prob=0.35)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                1, int(n_conv_chs), (1, time_conv_size), padding=(0, pad_size)),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
            # nn.Dropout(dropout),
            nn.Conv2d(
                int(n_conv_chs), int(n_conv_chs), (1, time_conv_size),
                padding=(0, pad_size)),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
            # nn.Dropout(dropout)
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
        if self.n_channels > 1:
            x = self.BN(torch.transpose(torch.unsqueeze(x, dim=2), 2, 1))
            # x = self.BN(x)
            x = self.spatial_conv(x)
        # GRU
        x = torch.transpose(torch.squeeze(x, dim=2), dim0=2, dim1=1)
        x = self.gru(x)
        x = torch.transpose(torch.unsqueeze(x, dim=1), dim0=2, dim1=3)
        x = self.feature_extractor(x)
        return self.fc(x.flatten(start_dim=1))


class SleepStagerChambon2018_with_esn(nn.Module):
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
                 max_pool_size_s=0.125, n_classes=5, input_size_s=30,
                 dropout=0.4, args=None):
        super().__init__()



        time_conv_size = int(time_conv_size_s * sfreq)
        # max_pool_size = int(max_pool_size_s * sfreq)
        max_pool_size = 2
        input_size = int(input_size_s * sfreq)
        pad_size = time_conv_size // 2
        self.n_channels = n_channels
        self.args = args

        self.washout = 128 * torch.ones((self.args.batch_size, 1))
        self.input_size = self.output_size = self.n_channels
        self.hidden_size = 100
        self.fc_input_size = 640

        len_last_layer = self._len_last_layer(
            n_channels, input_size, max_pool_size, n_conv_chs)

        self.BN = nn.BatchNorm2d(1)

        if n_channels > 1:
            self.spatial_conv = nn.Conv2d(1, n_channels, (n_channels, 1))

        if torch.cuda.is_available():
            self.esn = ESN(self.input_size, self.hidden_size, self.output_size).cuda()
            # self.gru = GRUModel(input_dim=n_channels, hidden_dim=n_channels, layer_dim=1, output_dim=n_channels, dropout_prob=0.35).cuda()
        else:
            self.esn = ESN(self.input_size, self.hidden_size, self.output_size)
            # self.gru = GRUModel(input_dim=n_channels, hidden_dim=n_channels, layer_dim=1, output_dim=n_channels,
            #                     dropout_prob=0.35)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                1, int(n_conv_chs), (1, time_conv_size), padding=(0, pad_size)),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
            # nn.Dropout(dropout),
            nn.Conv2d(
                int(n_conv_chs), int(n_conv_chs), (1, time_conv_size),
                padding=(0, pad_size)),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
            # nn.Dropout(dropout)
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            # nn.Linear(int(len_last_layer), n_classes)
            nn.Linear(self.fc_input_size, n_classes)
        )

        # self.fc1 = nn.Sequential(
        #     nn.Dropout(dropout),
        #     # nn.Linear(int(len_last_layer), n_classes)
        #     nn.Linear(self.fc_input_size, int(self.fc_input_size / 3))
        # )
        # self.fc2 = nn.Sequential(
        #     nn.Dropout(dropout),
        #     # nn.Linear(int(len_last_layer), n_classes)
        #     nn.Linear(int(self.fc_input_size / 3), max(int(self.fc_input_size / 27), 96))
        # )
        # self.fc3 = nn.Sequential(
        #     nn.Dropout(dropout),
        #     # nn.Linear(int(len_last_layer), n_classes)
        #     nn.Linear(max(int(self.fc_input_size / 27), 96), n_classes)
        # )


    @staticmethod
    def _len_last_layer(n_channels, input_size, max_pool_size, n_conv_chs):
        return n_channels * (input_size // (max_pool_size ** 2)) * n_conv_chs

    def forward(self, x, state='train', hidden=None, batch_size=None):
        """Forward pass.

        Parameters
        ---------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """
        if self.n_channels > 1:
            # x = self.BN(torch.transpose(torch.unsqueeze(x, dim=2), 2, 1))
            # x = self.BN(x)
            x = torch.transpose(torch.unsqueeze(x, dim=2), 2, 1)
            x = self.spatial_conv(x)
        # GRU
        x = torch.transpose(torch.squeeze(x, dim=2), dim0=2, dim1=1)
        if 'train' in state:
            self.esn.fit()
            output, hidden = self.esn(x, self.washout, h_0=hidden)
            # hidden = torch.mean(hidden.transpose(0, 1), dim=0, keepdim=True)
        elif 'test' in state:
            if batch_size is None:
                output, hidden = self.esn(x, torch.zeros((self.args.batch_size_val, 1)), h_0=hidden)
            else:
                output, hidden = self.esn(x, torch.zeros((batch_size, 1)), h_0=hidden)
        x = output
        x = torch.transpose(torch.unsqueeze(x, dim=1), dim0=2, dim1=3)
        x = self.feature_extractor(x)
        x = x.flatten(start_dim=1)
        try:
            return self.fc(x), hidden
        except:
            return self.fc(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # try:
        #     return self.fc3(x), hidden
        # except:
        #     return self.fc3(x)
        # try:
        #     return self.fc(x.flatten(start_dim=1)), hidden
        # except:
        #     return self.fc(x.flatten(start_dim=1))


class SleepStagerChambon2018_with_esn_EchoTorch(nn.Module):
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
                 max_pool_size_s=0.125, n_classes=5, input_size_s=30,
                 dropout=0.5, args=None):
        super().__init__()



        time_conv_size = int(time_conv_size_s * sfreq)
        # max_pool_size = int(max_pool_size_s * sfreq)
        max_pool_size = 2
        input_size = int(input_size_s * sfreq)
        pad_size = time_conv_size // 2
        self.n_channels = n_channels
        self.args = args

        self.washout = 128 * torch.ones((self.args.batch_size, 1))
        self.input_size = self.output_size = self.n_channels
        self.hidden_size = 100
        self.spectral_radius = 0.88
        self.leaky_rate = 0.9261
        reservoir_size = 410
        self.w_connectivity = 0.1954
        self.win_connectivity = 0.421
        self.wbias_connectivity = 0.333
        self.ridge_param = 0.00000409
        self.input_scaling = 0.9252
        self.bias_scaling = 0.079079


        len_last_layer = self._len_last_layer(
            n_channels, input_size, max_pool_size, n_conv_chs)

        self.BN = nn.BatchNorm2d(1)

        if n_channels > 1:
            self.spatial_conv = nn.Conv2d(1, n_channels, (n_channels, 1))


        if torch.cuda.is_available():
            self.esn = esn(input_dim=self.input_size, hidden_dim=self.hidden_size, output_dim=self.output_size, spectral_radius=self.spectral_radius\
            , leaky_rate=self.leaky_rate, w_connectivity=self.w_connectivity,\
            win_connectivity=self.win_connectivity, wbias_connectivity=self.wbias_connectivity,\
            input_scaling=self.input_scaling, bias_scaling=self.bias_scaling, ridge_param=self.ridge_param,\
            softmax_output=False, dtype=torch.float64).cuda()
            # self.gru = GRUModel(input_dim=n_channels, hidden_dim=n_channels, layer_dim=1, output_dim=n_channels, dropout_prob=0.35).cuda()
        else:
            self.esn = esn(input_dim=self.input_size, hidden_dim=self.hidden_size, output_dim=self.output_size, spectral_radius=self.spectral_radius\
            , leaky_rate=self.leaky_rate, w_connectivity=self.w_connectivity,\
            win_connectivity=self.win_connectivity, wbias_connectivity=self.wbias_connectivity,\
            input_scaling=self.input_scaling, bias_scaling=self.bias_scaling, ridge_param=self.ridge_param,\
            softmax_output=False, dtype=torch.float64)
            # self.gru = GRUModel(input_dim=n_channels, hidden_dim=n_channels, layer_dim=1, output_dim=n_channels,
            #                     dropout_prob=0.35)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                1, int(n_conv_chs), (1, time_conv_size), padding=(0, pad_size)),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
            # nn.Dropout(dropout),
            nn.Conv2d(
                int(n_conv_chs), int(n_conv_chs), (1, time_conv_size),
                padding=(0, pad_size)),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
            # nn.Dropout(dropout)
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            # nn.Linear(int(len_last_layer), n_classes)
            nn.Linear(1088, n_classes)
        )


    @staticmethod
    def _len_last_layer(n_channels, input_size, max_pool_size, n_conv_chs):
        return n_channels * (input_size // (max_pool_size ** 2)) * n_conv_chs

    def forward(self, x, state='train', hidden=None, batch_size=None):
        """Forward pass.

        Parameters
        ---------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """
        if self.n_channels > 1:
            # x = self.BN(torch.transpose(torch.unsqueeze(x, dim=2), 2, 1))
            # x = self.BN(x)
            x = torch.transpose(torch.unsqueeze(x, dim=2), 2, 1)
            x = self.spatial_conv(x)
        # GRU
        x = torch.transpose(torch.squeeze(x, dim=2), dim0=2, dim1=1)
        if 'train' in state:
            predict = self.esn(x)
            hidden = torch.mean(hidden.transpose(0, 1), dim=0, keepdim=True)
        elif 'test' in state:
            if batch_size is None:
                output, hidden = self.esn(x, torch.zeros((self.args.batch_size, 1)), hidden)
            else:
                output, hidden = self.esn(x, torch.zeros((batch_size, 1)), hidden)
            hidden = hidden.transpose(0, 1)
        x = output
        x = torch.transpose(torch.unsqueeze(x, dim=1), dim0=2, dim1=3)
        x = self.feature_extractor(x)
        try:
            return self.fc(x.flatten(start_dim=1)), hidden
        except:
            return self.fc(x.flatten(start_dim=1))


class SleepStagerChambon2018_ud(nn.Module):
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
                 max_pool_size_s=0.125, n_classes=5, input_size_s=30,
                 dropout=0.15):
        super().__init__()

        time_conv_size = int(time_conv_size_s * sfreq)
        # max_pool_size = int(max_pool_size_s * sfreq)
        max_pool_size = 2
        input_size = int(input_size_s * sfreq)
        pad_size = time_conv_size // 2
        self.n_channels = n_channels
        len_last_layer = self._len_last_layer(
            n_channels, input_size, max_pool_size, n_conv_chs)

        self.BN = nn.BatchNorm2d(1)
        self.IN = nn.InstanceNorm2d(1)

        if n_channels > 1:
            self.spatial_conv = nn.Conv2d(1, n_channels, (n_channels, 1))

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                1, int(n_conv_chs), (1, time_conv_size), padding=(0, pad_size)),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
            # nn.Dropout(dropout),
            nn.Conv2d(
                int(n_conv_chs), int(n_conv_chs), (1, time_conv_size),
                padding=(0, pad_size)),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
            # nn.Dropout(dropout)
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
        if self.n_channels > 1:
            # x = self.BN(self.IN(torch.transpose(torch.unsqueeze(x, dim=2), 2, 1)))
            x = self.IN(torch.transpose(torch.unsqueeze(x, dim=2), 2, 1))
            x = self.spatial_conv(x)
            x = x.transpose(1, 2)

        x = self.feature_extractor(x)
        return self.fc(x.flatten(start_dim=1))


class SleepStagerChambon2018_regression(nn.Module):
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

    def __init__(self, n_channels, sfreq, n_conv_chs=8, time_conv_size_s=0.5,
                 max_pool_size_s=0.125, n_classes=5, input_size_s=30,
                 dropout=0.25):
        super().__init__()

        time_conv_size = int(time_conv_size_s * sfreq)
        # max_pool_size = int(max_pool_size_s * sfreq)
        max_pool_size = 1
        input_size = int(input_size_s * sfreq)
        pad_size = time_conv_size // 2
        self.n_channels = n_channels
        len_last_layer = self._len_last_layer(
            n_channels, input_size, max_pool_size, n_conv_chs)

        if n_channels > 1:
            self.spatial_conv = nn.Conv2d(1, n_channels, (n_channels, 1))

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                1, n_conv_chs, (1, time_conv_size), padding=(0, pad_size)),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
            nn.Conv2d(
                n_conv_chs, n_conv_chs, (1, time_conv_size),
                padding=(0, pad_size)),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size))
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(len_last_layer, 1)
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
        if self.n_channels > 1:
            x = self.spatial_conv(torch.transpose(torch.unsqueeze(x, dim=2), 2, 1))
            x = x.transpose(1, 2)

        x = self.feature_extractor(x)
        return self.fc(x.flatten(start_dim=1))


class SleepStagerChambon2018_UD(nn.Module):
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

    def __init__(self, n_channels, sfreq, n_conv_chs=8, time_conv_size_s=0.5,
                 max_pool_size_s=0.125, n_classes=5, input_size_s=30,
                 dropout=0.25, multiply_factor_n_class=1):
        super().__init__()

        time_conv_size = int(time_conv_size_s * sfreq)
        # max_pool_size = int(max_pool_size_s * sfreq)
        max_pool_size = 1
        input_size = int(input_size_s * sfreq)
        pad_size = time_conv_size // 2
        self.n_channels = n_channels
        len_last_layer = self._len_last_layer(
            n_channels, input_size, max_pool_size, n_conv_chs)
        self.BN = nn.BatchNorm2d(1)
        if n_channels > 1:
            self.spatial_conv = nn.Conv2d(1, n_channels, (n_channels, 1))

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                1, n_conv_chs, (1, time_conv_size), padding=(0, pad_size)),
            nn.ReLU(),
            # nn.Dropout(0.65),
            nn.MaxPool2d((1, max_pool_size)),
            nn.Conv2d(
                n_conv_chs, n_conv_chs, (1, time_conv_size),
                padding=(0, pad_size)),
            nn.ReLU(),
            # nn.Dropout(0.65),
            nn.MaxPool2d((1, max_pool_size))
        )
        # self.fc = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(len_last_layer, multiply_factor_n_class*n_classes)
        # )

        self.classifier = nn.Sequential(nn.Dropout(0.8),
                                        nn.Linear(len_last_layer, 64),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.8),
                                        nn.Linear(128, 128),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.8))

        self.top_layer = nn.Linear(128, multiply_factor_n_class*n_classes)

    @staticmethod
    def _len_last_layer(n_channels, input_size, max_pool_size, n_conv_chs):
        return n_channels * (input_size // (max_pool_size ** 2)) * n_conv_chs * 6

    def forward(self, x):
        """Forward pass.

        Parameters
        ---------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """
        if self.n_channels > 1:
            # x = self.spatial_conv(self.BN(torch.transpose(torch.unsqueeze(x, dim=2), 2, 1)))
            x = self.spatial_conv(torch.transpose(torch.unsqueeze(x, dim=2), 2, 1))
            x = x.transpose(1, 2)

        x = self.feature_extractor(x)
        x = self.classifier(x.flatten(start_dim=1))
        if self.top_layer:
            x = self.top_layer(x)
        return x

class Pure_esn(nn.Module):
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
                 max_pool_size_s=0.125, n_classes=5, input_size_s=30,
                 dropout=0.4, args=None):
        super().__init__()

        self.n_channels = n_channels
        self.args = args

        self.washout = 32 * torch.ones((self.args.batch_size, 1))
        self.input_size = self.n_channels
        self.output_size = 1
        self.hidden_size = 100
        self.fc_input_size = 18


        self.BN = nn.BatchNorm2d(1)

        if torch.cuda.is_available():
            self.esn = ESN(self.input_size, self.hidden_size, self.output_size).cuda()
            # self.gru = GRUModel(input_dim=n_channels, hidden_dim=n_channels, layer_dim=1, output_dim=n_channels, dropout_prob=0.35).cuda()
        else:
            self.esn = ESN(self.input_size, self.hidden_size, self.output_size)
            # self.gru = GRUModel(input_dim=n_channels, hidden_dim=n_channels, layer_dim=1, output_dim=n_channels,
            #                     dropout_prob=0.35)

        self.fc = nn.Linear(in_features=self.fc_input_size, out_features=n_classes)
    def forward(self, x, state='train', hidden=None, batch_size=None):
        """Forward pass.

        Parameters
        ---------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """

        if 'train' in state:
            self.esn.fit()
            esn_output, hidden = self.esn(torch.transpose(x, dim0=2, dim1=1), self.washout, h_0=hidden)
            # hidden = torch.mean(hidden.transpose(0, 1), dim=0, keepdim=True)
        elif 'test' in state:
            if batch_size is None:
                esn_output, hidden = self.esn(torch.transpose(x, dim0=2, dim1=1), torch.zeros((self.args.batch_size, 1)), h_0=hidden)
            else:
                esn_output, hidden = self.esn(torch.transpose(x, dim0=2, dim1=1), torch.zeros((batch_size, 1)), h_0=hidden)

        x = esn_output.flatten(start_dim=1)
        try:
            return self.fc(x), hidden
        except:
            return self.fc(x)
        # try:
        #     return self.fc(x.flatten(start_dim=1)), hidden
        # except:
        #     return self.fc(x.flatten(start_dim=1))



def _do_train(model, loader, optimizer, criterion, device, metric, args=None):
    confidence = 0
    # training loop
    model.train()
    train_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []

    for idx_batch, (batch_x, batch_y) in enumerate(loader):
        optimizer.zero_grad()
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.int64)

        output = model(batch_x)
        confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
        confidence += torch.mean(confi)
        try:
            loss = criterion(output, batch_y)
        except:
            loss = criterion(output, torch.max(batch_y, 1)[0])
        # loss = my_loss(output, batch_y)
        # cur_analog_res = [output, batch_y]
        # analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
        # analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()

        loss.backward()
        optimizer.step()

        y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
        y_true_all.append(batch_y.cpu().numpy())

        train_loss[idx_batch] = loss.item()

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader)
    try:
        return np.mean(train_loss), perf, acc, confidence, analog_res
    except:
        return np.mean(train_loss), perf, acc, confidence

def _do_train_domain_adaptation(model, loader_tain, loader_val, optimizer, criterion, device, metric, args=None):
    confidence = 0
    # training loop
    model.train()
    train_loss = np.zeros(len(loader_tain))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader_tain) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []
    for val_idx_batch, (batch_x_val, batch_y_val) in enumerate(loader_val):
        batch_x_val = batch_x_val.to(device=device, dtype=torch.float32)
        batch_y_val = batch_y_val.to(device=device, dtype=torch.int64)
        for idx_batch, (batch_x, batch_y) in enumerate(loader_tain):
            optimizer.zero_grad()
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.int64)

            output, features = model(batch_x)
            output_val, features_val = model(batch_x_val)
            confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
            confidence += torch.mean(confi)
            try:
                cls_loss = criterion(output, batch_y)
            except:
                cls_loss = criterion(output, torch.max(batch_y, 1)[0])
            # loss = my_loss(output, batch_y)
            # cur_analog_res = [output, batch_y]
            # analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
            # analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()
            mmd_loss = mmd_linear(features, features_val)

            loss = cls_loss + mmd_loss

            loss.backward()
            optimizer.step()

            y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
            y_true_all.append(batch_y.cpu().numpy())

            train_loss[idx_batch] = loss.item()

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader_tain)
    try:
        return np.mean(train_loss), perf, acc, confidence, analog_res
    except:
        return np.mean(train_loss), perf, acc, confidence

def _do_train_fusion_big_DS(model, loader, optimizer, criterion, device, metric, args=None):
    confidence = 0
    # training loop
    model.train()
    train_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []

    for idx_batch, (batch_x, batch_y) in enumerate(loader):
        optimizer.zero_grad()
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.int64)
        try:
            output = model(batch_x, mode='teacher')
        except:
            output = model(batch_x)
        confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
        confidence += torch.mean(confi)
        try:
            loss = criterion(output, batch_y)
        except:
            loss = criterion(output, torch.max(batch_y, 1)[0])
        # loss = my_loss(output, batch_y)
        # cur_analog_res = [output, batch_y]
        # analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
        # analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()

        loss.backward()
        optimizer.step()

        y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
        y_true_all.append(batch_y.cpu().numpy())

        train_loss[idx_batch] = loss.item()

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader)
    try:
        return np.mean(train_loss), perf, acc, confidence, analog_res
    except:
        return np.mean(train_loss), perf, acc, confidence

def _do_train_fusion_EEGNet(model, loader, optimizer, criterion, device, metric, args=None):
    confidence = 0
    # training loop
    model.train()
    train_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []

    for idx_batch, (batch_x, batch_y) in enumerate(loader):
        optimizer.zero_grad()
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.int64)
        output = model(batch_x)

        confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
        confidence += torch.mean(confi)
        try:
            loss = criterion(output, batch_y)
        except:
            loss = criterion(output, torch.max(batch_y, 1)[0])
        # loss = my_loss(output, batch_y)
        # cur_analog_res = [output, batch_y]
        # analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
        # analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()

        loss.backward()
        optimizer.step()

        y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
        y_true_all.append(batch_y.cpu().numpy())

        train_loss[idx_batch] = loss.item()

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader)
    try:
        return np.mean(train_loss), perf, acc, confidence, analog_res
    except:
        return np.mean(train_loss), perf, acc, confidence

def _do_train_fusion_our_target_DS(model, loader, optimizer, criterion, device, metric, args=None):
    confidence = 0
    # training loop
    model.train()
    train_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []

    for idx_batch, (batch_x, batch_y) in enumerate(loader):
        optimizer.zero_grad()
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.int64)
        try:
            output = model(batch_x, mode='student')
        except:
            output = model(batch_x)
        confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
        confidence += torch.mean(confi)
        try:
            loss = criterion(output, batch_y)
        except:
            loss = criterion(output, torch.max(batch_y, 1)[0])
        # loss = my_loss(output, batch_y)
        # cur_analog_res = [output, batch_y]
        # analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
        # analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()

        loss.backward()
        optimizer.step()

        y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
        y_true_all.append(batch_y.cpu().numpy())

        train_loss[idx_batch] = loss.item()

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader)
    try:
        return np.mean(train_loss), perf, acc, confidence, analog_res
    except:
        return np.mean(train_loss), perf, acc, confidence

def _do_train_without_transform_after_encoder(model, model_enc, loader, optimizer, criterion, device, metric, args=None):
    confidence = 0
    # training loop
    model.train()
    train_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []

    checkpoint = torch.load('C:\\Users\\Avishay David Malka\\Work\\AI_dsp\\dsp\\src\\sim_Encoder.pt')
    model_enc.load_state_dict(checkpoint['model_state_dict'])
    model_enc.eval()
    if torch.cuda.is_available():
        model_enc.cuda()

    for idx_batch, (batch_x, batch_y) in enumerate(loader):
        batch_x_after_enc = torch.zeros((batch_x.shape[0], batch_x.shape[1], model_enc.encoder[-1].out_features), requires_grad=True).to(device=device, dtype=torch.float32)
        optimizer.zero_grad()
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.int64)

        batch_std, batch_mu = torch.std_mean(batch_x, dim=(0, 2))

        batch_x = normalized_batch(batch_x=batch_x, std=batch_std, mu=batch_mu)

        with torch.no_grad():
            for ch in range(batch_x.shape[1]):
                batch_x_after_enc[:, ch, :] = model_enc(batch_x[:, ch, :])


        output = model(batch_x_after_enc)
        confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
        confidence += torch.mean(confi)
        try:
            loss = criterion(output, batch_y)
        except:
            loss = criterion(output, torch.max(batch_y, 1)[0])
        # loss = my_loss(output, batch_y)
        # cur_analog_res = [output, batch_y]
        # analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
        # analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()

        loss.backward()
        optimizer.step()

        y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
        y_true_all.append(batch_y.cpu().numpy())

        train_loss[idx_batch] = loss.item()

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader)
    try:
        return np.mean(train_loss), perf, acc, confidence, analog_res
    except:
        return np.mean(train_loss), perf, acc, confidence

def _do_train_without_transform(model, loader, optimizer, criterion, device, metric, args=None):
    confidence = 0
    # training loop
    model.train()
    train_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []


    for idx_batch, (batch_x, batch_y) in enumerate(loader):
        optimizer.zero_grad()
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.int64)

        batch_std, batch_mu = torch.std_mean(batch_x, dim=(0, 2))

        batch_x = normalized_batch(batch_x=batch_x, std=batch_std, mu=batch_mu)

        output = model(batch_x)
        confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
        confidence += torch.mean(confi)
        try:
            loss = criterion(output, batch_y)
        except:
            loss = criterion(output, torch.max(batch_y, 1)[0])
        # loss = my_loss(output, batch_y)
        # cur_analog_res = [output, batch_y]
        # analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
        # analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()

        loss.backward()
        optimizer.step()

        y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
        y_true_all.append(batch_y.cpu().numpy())

        train_loss[idx_batch] = loss.item()

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader)
    try:
        return np.mean(train_loss), perf, acc, confidence, analog_res
    except:
        return np.mean(train_loss), perf, acc, confidence

def _do_train_without_transform_with_our_loss(model, loader, optimizer, criterion, device, metric, args=None):
    confidence = 0
    # training loop
    model.train()
    train_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []

    Our_loss = Soft_Nearset_Neighbor_Loss()
    for idx_batch, (batch_x, batch_y) in enumerate(loader):
        optimizer.zero_grad()
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.int64)
        # try:
        #     batch_std, batch_mu = torch.std_mean(batch_x, dim=(0, 2))
        #
        #     batch_x = normalized_batch(batch_x=batch_x, std=batch_std, mu=batch_mu)
        # except:
        #     pass

        output, logits = model(batch_x)
        confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
        confidence += torch.mean(confi)
        try:
            loss_reg = criterion(output, batch_y)
        except:
            loss_reg = criterion(output, torch.max(batch_y, 1)[0])
        # loss = my_loss(output, batch_y)
        # cur_analog_res = [output, batch_y]
        # analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
        # analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()
        # our_loss = NeuroBraveLoss(x=logits, labels=batch_y, mode='Mahalanobis')
        our_loss = Our_loss.forward(logits, batch_y)

        loss = loss_reg + our_loss
        # loss = loss_reg

        loss.backward()
        optimizer.step()

        y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
        y_true_all.append(batch_y.cpu().numpy())

        train_loss[idx_batch] = loss.item()

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader)
    try:
        return np.mean(train_loss), perf, acc, confidence, analog_res
    except:
        return np.mean(train_loss), perf, acc, confidence

def _validate(model, loader, criterion, device, metric, args=None):
    confidence = 0
    # validation loop
    model.eval()

    val_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []
    with torch.no_grad():
        for idx_batch, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.int64)
            output = model.forward(batch_x)

            confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
            confidence += torch.mean(confi)
            try:
                loss = criterion(output, batch_y)
            except:
                loss = criterion(output, torch.max(batch_y, 1)[0])
            # loss = my_loss(output, batch_y)
            # cur_analog_res = [output, batch_y]
            # analog_res.append(cur_analog_res)
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()
            val_loss[idx_batch] = loss.item()

            y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
            y_true_all.append(batch_y.cpu().numpy())

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader)

    try:
        return np.mean(val_loss), perf, acc, confidence, analog_res
    except:
        return np.mean(val_loss), perf, acc, confidence


def _validate_with_majority(model, loader, criterion, device, metric, args=None):
    confidence = 0
    # validation loop
    model.eval()

    val_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []
    with torch.no_grad():
        for idx_batch, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.int64)

            output = model.forward(batch_x)

            confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
            confidence += torch.mean(confi)
            try:
                loss = criterion(output, batch_y)
            except:
                loss = criterion(output, torch.max(batch_y, 1)[0])
            # loss = my_loss(output, batch_y)
            # cur_analog_res = [output, batch_y]
            # analog_res.append(cur_analog_res)
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()
            val_loss[idx_batch] = loss.item()

            decisions = torch.argmax(output, axis=1)

            majority_decisions = torch.zeros_like(decisions)

            values, indices = torch.mode(decisions)
            majority_decisions[:] = values
            y_pred_all.append(majority_decisions.cpu().numpy())
            y_true_all.append(batch_y.cpu().numpy())

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader)

    try:
        return np.mean(val_loss), perf, acc, confidence, analog_res
    except:
        return np.mean(val_loss), perf, acc, confidence


def _validate_with_majority_and_regular(model, loader, criterion, device, metric, args=None):
    confidence = 0
    # validation loop
    model.eval()

    val_loss = np.zeros(len(loader))
    y_pred_all_majority, y_pred_all_regular, y_true_all = list(), list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []
    with torch.no_grad():
        for idx_batch, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.int64)

            output = model.forward(batch_x)

            confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
            confidence += torch.mean(confi)
            try:
                loss = criterion(output, batch_y)
            except:
                loss = criterion(output, torch.max(batch_y, 1)[0])
            # loss = my_loss(output, batch_y)
            # cur_analog_res = [output, batch_y]
            # analog_res.append(cur_analog_res)
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()
            val_loss[idx_batch] = loss.item()

            decisions = torch.argmax(output, axis=1)

            majority_decisions = torch.zeros_like(decisions)

            values, indices = torch.mode(decisions)
            majority_decisions[:] = values
            y_pred_all_majority.append(majority_decisions.cpu().numpy())
            y_pred_all_regular.append(decisions.cpu().numpy())
            y_true_all.append(batch_y.cpu().numpy())

    if 0:
        plt.figure()
        plt.plot(range(len(val_loss)), val_loss)
        relax_factor = torch.sum(loader.dataset.y == 0) / len(loader.dataset.y)
        neutral_factor = torch.sum(loader.dataset.y == 1) / len(loader.dataset.y)
        focus_factor = torch.sum(loader.dataset.y == 2) / len(loader.dataset.y)
        plt.axvspan(0, int(relax_factor * len(loader)), facecolor='green', alpha=0.5)
        plt.axvspan(int(relax_factor * len(loader)), int(relax_factor * len(loader) + neutral_factor * len(loader)),
                    facecolor='yellow', alpha=0.5)
        plt.axvspan(int(relax_factor * len(loader) + neutral_factor * len(loader)), len(loader), facecolor='red', alpha=0.5)
        plt.title('Validation loss over batches [in batch size resolution]')


    y_pred_regular = np.concatenate(y_pred_all_regular)
    y_pred_majority = np.concatenate(y_pred_all_majority)
    y_true = np.concatenate(y_true_all)
    perf_regular = metric(y_true, y_pred_regular)
    perf_majority = metric(y_true, y_pred_majority)
    acc_regular = np.mean((y_pred_regular == y_true))
    acc_majority = np.mean((y_pred_majority == y_true))
    confidence /= len(loader)

    try:
        return np.mean(val_loss), perf_regular, perf_majority, acc_regular, acc_majority, confidence, analog_res
    except:
        return np.mean(val_loss), perf, acc, confidence

def _validate_with_majority_and_regular_fusion_big_DS(model, loader, criterion, device, metric, args=None):
    confidence = 0
    # validation loop
    model.eval()

    val_loss = np.zeros(len(loader))
    y_pred_all_majority, y_pred_all_regular, y_true_all = list(), list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []
    with torch.no_grad():
        for idx_batch, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.int64)
            try:
                output = model.forward(batch_x, mode='teacher')
            except:
                output = model(batch_x)

            confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
            confidence += torch.mean(confi)
            try:
                loss = criterion(output, batch_y)
            except:
                loss = criterion(output, torch.max(batch_y, 1)[0])
            # loss = my_loss(output, batch_y)
            # cur_analog_res = [output, batch_y]
            # analog_res.append(cur_analog_res)
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()
            val_loss[idx_batch] = loss.item()

            decisions = torch.argmax(output, axis=1)

            majority_decisions = torch.zeros_like(decisions)

            values, indices = torch.mode(decisions)
            majority_decisions[:] = values
            y_pred_all_majority.append(majority_decisions.cpu().numpy())
            y_pred_all_regular.append(decisions.cpu().numpy())
            y_true_all.append(batch_y.cpu().numpy())

    if 0:
        plt.figure()
        plt.plot(range(len(val_loss)), val_loss)
        relax_factor = torch.sum(loader.dataset.y == 0) / len(loader.dataset.y)
        neutral_factor = torch.sum(loader.dataset.y == 1) / len(loader.dataset.y)
        focus_factor = torch.sum(loader.dataset.y == 2) / len(loader.dataset.y)
        plt.axvspan(0, int(relax_factor * len(loader)), facecolor='green', alpha=0.5)
        plt.axvspan(int(relax_factor * len(loader)), int(relax_factor * len(loader) + neutral_factor * len(loader)),
                    facecolor='yellow', alpha=0.5)
        plt.axvspan(int(relax_factor * len(loader) + neutral_factor * len(loader)), len(loader), facecolor='red', alpha=0.5)
        plt.title('Validation loss over batches [in batch size resolution]')


    y_pred_regular = np.concatenate(y_pred_all_regular)
    y_pred_majority = np.concatenate(y_pred_all_majority)
    y_true = np.concatenate(y_true_all)
    perf_regular = metric(y_true, y_pred_regular)
    perf_majority = metric(y_true, y_pred_majority)
    acc_regular = np.mean((y_pred_regular == y_true))
    acc_majority = np.mean((y_pred_majority == y_true))
    confidence /= len(loader)

    try:
        return np.mean(val_loss), perf_regular, perf_majority, acc_regular, acc_majority, confidence, analog_res
    except:
        return np.mean(val_loss), perf, acc, confidence

def _validate_with_majority_and_regular_fusion_EEGNet(model, loader, criterion, device, metric, args=None):
    confidence = 0
    # validation loop
    model.eval()

    val_loss = np.zeros(len(loader))
    y_pred_all_majority, y_pred_all_regular, y_true_all = list(), list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []
    with torch.no_grad():
        for idx_batch, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.int64)
            output = model(batch_x)


            confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
            confidence += torch.mean(confi)
            try:
                loss = criterion(output, batch_y)
            except:
                loss = criterion(output, torch.max(batch_y, 1)[0])
            # loss = my_loss(output, batch_y)
            # cur_analog_res = [output, batch_y]
            # analog_res.append(cur_analog_res)
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()
            val_loss[idx_batch] = loss.item()

            decisions = torch.argmax(output, axis=1)

            majority_decisions = torch.zeros_like(decisions)

            values, indices = torch.mode(decisions)
            majority_decisions[:] = values
            y_pred_all_majority.append(majority_decisions.cpu().numpy())
            y_pred_all_regular.append(decisions.cpu().numpy())
            y_true_all.append(batch_y.cpu().numpy())

    if 0:
        plt.figure()
        plt.plot(range(len(val_loss)), val_loss)
        relax_factor = torch.sum(loader.dataset.y == 0) / len(loader.dataset.y)
        neutral_factor = torch.sum(loader.dataset.y == 1) / len(loader.dataset.y)
        focus_factor = torch.sum(loader.dataset.y == 2) / len(loader.dataset.y)
        plt.axvspan(0, int(relax_factor * len(loader)), facecolor='green', alpha=0.5)
        plt.axvspan(int(relax_factor * len(loader)), int(relax_factor * len(loader) + neutral_factor * len(loader)),
                    facecolor='yellow', alpha=0.5)
        plt.axvspan(int(relax_factor * len(loader) + neutral_factor * len(loader)), len(loader), facecolor='red', alpha=0.5)
        plt.title('Validation loss over batches [in batch size resolution]')


    y_pred_regular = np.concatenate(y_pred_all_regular)
    y_pred_majority = np.concatenate(y_pred_all_majority)
    y_true = np.concatenate(y_true_all)
    perf_regular = metric(y_true, y_pred_regular)
    perf_majority = metric(y_true, y_pred_majority)
    acc_regular = np.mean((y_pred_regular == y_true))
    acc_majority = np.mean((y_pred_majority == y_true))
    confidence /= len(loader)

    try:
        return np.mean(val_loss), perf_regular, perf_majority, acc_regular, acc_majority, confidence, analog_res
    except:
        return np.mean(val_loss), perf, acc, confidence

def _validate_with_majority_and_regular_fusion_our_target_DS(model, loader, criterion, device, metric, args=None):
    confidence = 0
    # validation loop
    model.eval()

    val_loss = np.zeros(len(loader))
    y_pred_all_majority, y_pred_all_regular, y_true_all = list(), list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []
    with torch.no_grad():
        for idx_batch, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.int64)
            try:
                output = model.forward(batch_x, mode='student')
            except:
                output = model(batch_x)

            confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
            confidence += torch.mean(confi)
            try:
                loss = criterion(output, batch_y)
            except:
                loss = criterion(output, torch.max(batch_y, 1)[0])
            # loss = my_loss(output, batch_y)
            # cur_analog_res = [output, batch_y]
            # analog_res.append(cur_analog_res)
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()
            val_loss[idx_batch] = loss.item()

            decisions = torch.argmax(output, axis=1)

            majority_decisions = torch.zeros_like(decisions)

            values, indices = torch.mode(decisions)
            majority_decisions[:] = values
            y_pred_all_majority.append(majority_decisions.cpu().numpy())
            y_pred_all_regular.append(decisions.cpu().numpy())
            y_true_all.append(batch_y.cpu().numpy())

    if 0:
        plt.figure()
        plt.plot(range(len(val_loss)), val_loss)
        relax_factor = torch.sum(loader.dataset.y == 0) / len(loader.dataset.y)
        neutral_factor = torch.sum(loader.dataset.y == 1) / len(loader.dataset.y)
        focus_factor = torch.sum(loader.dataset.y == 2) / len(loader.dataset.y)
        plt.axvspan(0, int(relax_factor * len(loader)), facecolor='green', alpha=0.5)
        plt.axvspan(int(relax_factor * len(loader)), int(relax_factor * len(loader) + neutral_factor * len(loader)),
                    facecolor='yellow', alpha=0.5)
        plt.axvspan(int(relax_factor * len(loader) + neutral_factor * len(loader)), len(loader), facecolor='red', alpha=0.5)
        plt.title('Validation loss over batches [in batch size resolution]')


    y_pred_regular = np.concatenate(y_pred_all_regular)
    y_pred_majority = np.concatenate(y_pred_all_majority)
    y_true = np.concatenate(y_true_all)
    perf_regular = metric(y_true, y_pred_regular)
    perf_majority = metric(y_true, y_pred_majority)
    acc_regular = np.mean((y_pred_regular == y_true))
    acc_majority = np.mean((y_pred_majority == y_true))
    confidence /= len(loader)

    try:
        return np.mean(val_loss), perf_regular, perf_majority, acc_regular, acc_majority, confidence, analog_res
    except:
        return np.mean(val_loss), perf, acc, confidence

def _validate_with_majority_and_regular_without_transform_with_our_loss(model, loader, criterion, device, metric, args=None):
    confidence = 0
    # validation loop
    model.eval()

    val_loss = np.zeros(len(loader))
    y_pred_all_majority, y_pred_all_regular, y_true_all = list(), list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []
    with torch.no_grad():
        Our_loss = Soft_Nearset_Neighbor_Loss()
        for idx_batch, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.int64)
            # try:
            #     batch_std, batch_mu = torch.std_mean(batch_x, dim=(0, 2))
            #
            #     batch_x = normalized_batch(batch_x=batch_x, std=batch_std, mu=batch_mu)
            # except:
            #     pass

            output, logits = model.forward(batch_x)

            confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
            confidence += torch.mean(confi)
            try:
                loss_reg = criterion(output, batch_y)
            except:
                loss_reg = criterion(output, torch.max(batch_y, 1)[0])
            # loss = my_loss(output, batch_y)
            # cur_analog_res = [output, batch_y]
            # analog_res.append(cur_analog_res)
            # our_loss = NeuroBraveLoss(x=logits, labels=batch_y, mode='Mahalanobis')
            our_loss = Our_loss.forward(logits, batch_y)

            loss = loss_reg + our_loss
            # loss = loss_reg

            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()
            val_loss[idx_batch] = loss.item()

            decisions = torch.argmax(output, axis=1)

            majority_decisions = torch.zeros_like(decisions)

            values, indices = torch.mode(decisions)
            majority_decisions[:] = values
            y_pred_all_majority.append(majority_decisions.cpu().numpy())
            y_pred_all_regular.append(decisions.cpu().numpy())
            y_true_all.append(batch_y.cpu().numpy())

    if 0:
        plt.figure()
        plt.plot(range(len(val_loss)), val_loss)
        relax_factor = torch.sum(loader.dataset.y == 0) / len(loader.dataset.y)
        neutral_factor = torch.sum(loader.dataset.y == 1) / len(loader.dataset.y)
        focus_factor = torch.sum(loader.dataset.y == 2) / len(loader.dataset.y)
        plt.axvspan(0, int(relax_factor * len(loader)), facecolor='green', alpha=0.5)
        plt.axvspan(int(relax_factor * len(loader)), int(relax_factor * len(loader) + neutral_factor * len(loader)),
                    facecolor='yellow', alpha=0.5)
        plt.axvspan(int(relax_factor * len(loader) + neutral_factor * len(loader)), len(loader), facecolor='red', alpha=0.5)
        plt.title('Validation loss over batches [in batch size resolution]')


    y_pred_regular = np.concatenate(y_pred_all_regular)
    y_pred_majority = np.concatenate(y_pred_all_majority)
    y_true = np.concatenate(y_true_all)
    perf_regular = metric(y_true, y_pred_regular)
    perf_majority = metric(y_true, y_pred_majority)
    acc_regular = np.mean((y_pred_regular == y_true))
    acc_majority = np.mean((y_pred_majority == y_true))
    confidence /= len(loader)

    try:
        return np.mean(val_loss), perf_regular, perf_majority, acc_regular, acc_majority, confidence, analog_res
    except:
        return np.mean(val_loss), perf, acc, confidence

def _validate_with_majority_and_regular_without_transform_after_encoder(model, model_enc, loader, criterion, device, metric, args=None):
    confidence = 0
    # validation loop
    model.eval()

    checkpoint = torch.load('C:\\Users\\Avishay David Malka\\Work\\AI_dsp\\dsp\\src\\sim_Encoder.pt')
    model_enc.load_state_dict(checkpoint['model_state_dict'])
    model_enc.eval()

    if torch.cuda.is_available():
        model_enc.cuda()

    val_loss = np.zeros(len(loader))
    y_pred_all_majority, y_pred_all_regular, y_true_all = list(), list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []
    with torch.no_grad():
        for idx_batch, (batch_x, batch_y) in enumerate(loader):
            batch_x_after_enc = torch.zeros(
                (batch_x.shape[0], batch_x.shape[1], model_enc.encoder[-1].out_features), requires_grad=True).to(
                device=device, dtype=torch.float32)
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.int64)

            batch_std, batch_mu = torch.std_mean(batch_x, dim=(0, 2))

            batch_x = normalized_batch(batch_x=batch_x, std=batch_std, mu=batch_mu)

            with torch.no_grad():
                for ch in range(batch_x.shape[1]):
                    batch_x_after_enc[:, ch, :] = model_enc(batch_x[:, ch, :])

            output = model.forward(batch_x)

            confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
            confidence += torch.mean(confi)
            try:
                loss = criterion(output, batch_y)
            except:
                loss = criterion(output, torch.max(batch_y, 1)[0])
            # loss = my_loss(output, batch_y)
            # cur_analog_res = [output, batch_y]
            # analog_res.append(cur_analog_res)
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()
            val_loss[idx_batch] = loss.item()

            decisions = torch.argmax(output, axis=1)

            majority_decisions = torch.zeros_like(decisions)

            values, indices = torch.mode(decisions)
            majority_decisions[:] = values
            y_pred_all_majority.append(majority_decisions.cpu().numpy())
            y_pred_all_regular.append(decisions.cpu().numpy())
            y_true_all.append(batch_y.cpu().numpy())

    if 0:
        plt.figure()
        plt.plot(range(len(val_loss)), val_loss)
        relax_factor = torch.sum(loader.dataset.y == 0) / len(loader.dataset.y)
        neutral_factor = torch.sum(loader.dataset.y == 1) / len(loader.dataset.y)
        focus_factor = torch.sum(loader.dataset.y == 2) / len(loader.dataset.y)
        plt.axvspan(0, int(relax_factor * len(loader)), facecolor='green', alpha=0.5)
        plt.axvspan(int(relax_factor * len(loader)), int(relax_factor * len(loader) + neutral_factor * len(loader)),
                    facecolor='yellow', alpha=0.5)
        plt.axvspan(int(relax_factor * len(loader) + neutral_factor * len(loader)), len(loader), facecolor='red', alpha=0.5)
        plt.title('Validation loss over batches [in batch size resolution]')


    y_pred_regular = np.concatenate(y_pred_all_regular)
    y_pred_majority = np.concatenate(y_pred_all_majority)
    y_true = np.concatenate(y_true_all)
    perf_regular = metric(y_true, y_pred_regular)
    perf_majority = metric(y_true, y_pred_majority)
    acc_regular = np.mean((y_pred_regular == y_true))
    acc_majority = np.mean((y_pred_majority == y_true))
    confidence /= len(loader)

    try:
        return np.mean(val_loss), perf_regular, perf_majority, acc_regular, acc_majority, confidence, analog_res
    except:
        return np.mean(val_loss), perf, acc, confidence

def _do_train_lstm(model, loader, optimizer, criterion, device, metric, args=None, hn=None, cn=None):
    confidence = 0
    # training loop
    model.train()
    train_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []

    for idx_batch, (batch_x, batch_y) in enumerate(loader):
        optimizer.zero_grad()
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.int64)
        try:
            output, hn, cn = model(batch_x, hn=hn, cn=cn)
        except:
            output, _, _ = model(batch_x, hn=hn, cn=cn)
        confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
        confidence += torch.mean(confi)
        try:
            loss = criterion(output, batch_y)
        except:
            loss = criterion(output, torch.max(batch_y, 1)[0])
        # loss = my_loss(output, batch_y)
        # cur_analog_res = [output, batch_y]
        # analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
        # analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()

        loss.backward()
        optimizer.step()

        y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
        y_true_all.append(batch_y.cpu().numpy())

        train_loss[idx_batch] = loss.item()

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader)
    try:
        return np.mean(train_loss), perf, acc, confidence, analog_res, hn, cn
    except:
        return np.mean(train_loss), perf, acc, confidence

def _validate_lstm(model, loader, criterion, device, metric, args=None, hn=None, cn=None):
    confidence = 0
    # validation loop
    model.eval()

    val_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []
    with torch.no_grad():
        for idx_batch, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.int64)
            output, hn, cn = model.forward(batch_x, hn=None, cn=None)

            confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
            confidence += torch.mean(confi)
            try:
                loss = criterion(output, batch_y)
            except:
                loss = criterion(output, torch.max(batch_y, 1)[0])
            # loss = my_loss(output, batch_y)
            # cur_analog_res = [output, batch_y]
            # analog_res.append(cur_analog_res)
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()
            val_loss[idx_batch] = loss.item()

            y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
            y_true_all.append(batch_y.cpu().numpy())

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader)

    try:
        return np.mean(val_loss), perf, acc, confidence, analog_res
    except:
        return np.mean(val_loss), perf, acc, confidence

def _do_train_AttenSleep(model, loader, optimizer, criterion, device, metric, args=None):
    confidence = 0
    # training loop
    model.train()
    train_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []

    for idx_batch, (batch_x, batch_y) in enumerate(loader):
        optimizer.zero_grad()
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.int64)
        confidence_tmp = 0
        loss = 0
        output_tot = 0
        for ch in range(batch_x.shape[1]):
            output = model(torch.unsqueeze(batch_x[:, ch, :], dim=1))
            confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
            confidence_tmp += confi

            output_tot += torch.nn.functional.softmax(output, dim=1) * output

        try:
            loss += criterion(output_tot, batch_y)
        except:
            loss += criterion(output_tot, torch.max(batch_y, 1)[0])
        confidence += torch.mean(confidence_tmp / batch_x.shape[1])
        # loss = my_loss(output, batch_y)
        # cur_analog_res = [output, batch_y]
        # analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
        # analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()

        loss.backward()
        optimizer.step()

        y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
        y_true_all.append(batch_y.cpu().numpy())

        train_loss[idx_batch] = loss.item()

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader)
    try:
        return np.mean(train_loss), perf, acc, confidence, analog_res
    except:
        return np.mean(train_loss), perf, acc, confidence

def _validate_AttenSleep(model, loader, criterion, device, metric, args=None):
    confidence = 0
    # validation loop
    model.eval()

    val_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []
    with torch.no_grad():
        for idx_batch, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.int64)

            confidence_tmp = 0
            loss = 0
            output_tot = 0
            for ch in range(batch_x.shape[1]):
                output = model.forward(torch.unsqueeze(batch_x[:, ch, :], dim=1))
                confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
                confidence_tmp += confi

                output_tot += torch.nn.functional.softmax(output, dim=1) * output
            try:
                loss += criterion(output_tot, batch_y)
            except:
                loss += criterion(output_tot, torch.max(batch_y, 1)[0])

            confidence += torch.mean(confidence_tmp / batch_x.shape[1])
            # loss = my_loss(output, batch_y)
            # cur_analog_res = [output, batch_y]
            # analog_res.append(cur_analog_res)
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()
            val_loss[idx_batch] = loss.item()

            y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
            y_true_all.append(batch_y.cpu().numpy())

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader)

    try:
        return np.mean(val_loss), perf, acc, confidence, analog_res
    except:
        return np.mean(val_loss), perf, acc, confidence

def _do_train_VAE(model, loader, optimizer, criterion, device, metric, args=None):
    confidence = 0
    # training loop
    model.train()
    train_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    for idx_batch, (batch_x, batch_y) in enumerate(loader):
        optimizer.zero_grad()
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.int64)

        recon_data, mu, logvar, output = model(batch_x)
        loss_vae, bce, kld = loss_fn(recon_data, batch_x, mu, logvar)

        # output = model(batch_x)
        confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
        confidence += torch.mean(confi)
        try:
            loss_classification = criterion(output, batch_y)
        except:
            loss_classification = criterion(output, torch.max(batch_y, 1)[0])
        # loss = my_loss(output, batch_y)
        # cur_analog_res = [output, batch_y]
        # analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
        # analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()
        loss = loss_classification + loss_vae
        loss.backward()
        optimizer.step()

        y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
        y_true_all.append(batch_y.cpu().numpy())

        train_loss[idx_batch] = loss.item()

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader)
    try:
        return np.mean(train_loss), perf, acc, confidence, analog_res
    except:
        return np.mean(train_loss), perf, acc, confidence

def _validate_VAE(model, loader, criterion, device, metric, args=None):
    confidence = 0
    # validation loop
    model.eval()

    val_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []
    with torch.no_grad():
        for idx_batch, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.int64)
            recon_data, mu, logvar, output = model.forward(batch_x)
            loss_vae, bce, kld = loss_fn(recon_data, batch_x, mu, logvar)

            confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
            confidence += torch.mean(confi)
            try:
                loss_classification = criterion(output, batch_y)
            except:
                loss_classification = criterion(output, torch.max(batch_y, 1)[0])
            # loss = my_loss(output, batch_y)
            # cur_analog_res = [output, batch_y]
            # analog_res.append(cur_analog_res)
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()

            loss = loss_vae + loss_classification
            val_loss[idx_batch] = loss.item()

            y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
            y_true_all.append(batch_y.cpu().numpy())

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader)

    try:
        return np.mean(val_loss), perf, acc, confidence, analog_res
    except:
        return np.mean(val_loss), perf, acc, confidence

def _do_train_subgroup_t_SNE(model, loader, tsne_loader, optimizer, criterion, device, metric, args=None):
    confidence = 0
    # training loop
    model.train()
    train_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []
    tsne_data = next(iter(tsne_loader))

    best_err_t_SNE = np.inf
    tsne = []
    for idx_batch, (batch_x, batch_y) in enumerate(loader):
        optimizer.zero_grad()
        # batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.int64)

        con_data = torch.cat((batch_x, tsne_data), dim=0).detach().numpy()

        if idx_batch == 0:
            tsne_results = np.zeros((con_data.shape[0], con_data.shape[1], args.t_SNE_n_components))
            for ch in range(tsne_results.shape[1]):
                tsne.append(TSNE(n_components=args.t_SNE_n_components, init='random', method='exact', verbose=0,
                            perplexity=40, n_iter=300))

        for ch in range(tsne_results.shape[1]):
            tsne_results[:, ch, :] = tsne[ch].fit_transform(con_data[:, ch, :])
        if idx_batch == (len(loader)-1):
            for ch in range(tsne_results.shape[1]):
                with open(f'{args.pkl_t_SNE_filename}_based_on_sub_group_size_of_{args.N_random_samples}_ch_{ch}.pkl', 'wb') as file:
                    pickle.dump(tsne[ch], file)

        batch_x = torch.from_numpy(tsne_results[0:batch_x.shape[0]]).to(device=device, dtype=torch.float32)

        output = model(batch_x)
        confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
        confidence += torch.mean(confi)
        try:
            loss = criterion(output, batch_y)
        except:
            loss = criterion(output, torch.max(batch_y, 1)[0])
        # loss = my_loss(output, batch_y)
        # cur_analog_res = [output, batch_y]
        analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
        analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()

        loss.backward()
        optimizer.step()

        y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
        y_true_all.append(batch_y.cpu().numpy())

        train_loss[idx_batch] = loss.item()

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader)
    try:
        return np.mean(train_loss), perf, acc, confidence, analog_res
    except:
        return np.mean(train_loss), perf, acc, confidence

def _validate_subgroup_t_SNE(model, loader, tsne_loader, criterion, device, metric, args=None):
    confidence = 0
    # validation loop
    model.eval()

    val_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []

    with open(f'{args.pkl_t_SNE_filename} + based_on_sub_group_size_of_{args.N_random_samples} + .pkl', 'rb') as file:
        tsne = pickle.load(file)

    with torch.no_grad():
        tsne_data = next(iter(tsne_loader))
        for idx_batch, (batch_x, batch_y) in enumerate(loader):
            # batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.int64)

            con_data = torch.cat((batch_x, tsne_data), dim=0).detach().numpy()
            if idx_batch == 0:
                tsne_results = np.zeros((con_data.shape[0], con_data.shape[1], args.t_SNE_n_components))

            for ch in range(tsne_results.shape[1]):
                tsne_results[:, ch, :] = tsne.fit_transform(con_data[:, ch, :])

            batch_x = torch.from_numpy(tsne_results[0:batch_x.shape[0]]).to(device=device, dtype=torch.float32)

            output = model.forward(batch_x)

            confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
            confidence += torch.mean(confi)
            try:
                loss = criterion(output, batch_y)
            except:
                loss = criterion(output, torch.max(batch_y, 1)[0])
            # loss = my_loss(output, batch_y)
            # cur_analog_res = [output, batch_y]
            # analog_res.append(cur_analog_res)
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()
            val_loss[idx_batch] = loss.item()

            y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
            y_true_all.append(batch_y.cpu().numpy())

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader)

    try:
        return np.mean(val_loss), perf, acc, confidence, analog_res
    except:
        return np.mean(val_loss), perf, acc, confidence


def _do_train_subgroup_t_SNE_with_torch(model, loader, tsne_loader, optimizer, criterion, device, metric, args=None):
    confidence = 0
    # training loop
    model.train()
    train_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []
    tsne_data = next(iter(tsne_loader))

    best_err_t_SNE = np.inf
    tsne = []
    bn = nn.BatchNorm1d(num_features=tsne_data.shape[1]).to(device=device)
    train_loss_agg=0
    for idx_batch, (batch_x, batch_y) in enumerate(loader):
        optimizer.zero_grad()
        # batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.int64)

        con_data_before_norm = torch.cat((batch_x, tsne_data), dim=0).to(device=device, dtype=torch.float32)
        con_data = bn(con_data_before_norm)

        if idx_batch == 0:

            tsne_results = np.zeros((con_data.shape[0], con_data.shape[1], args.t_SNE_n_components))
            for ch in range(tsne_results.shape[1]):
                tsne.append(TSNE(n_components=args.t_SNE_n_components, verbose=False,
                            perplexity=40, n_iter=300, initial_dims=args.raw_time_duration))

        for ch in range(tsne_results.shape[1]):
            tsne_results[:, ch, :] = tsne[ch].fit_transform(con_data[:, ch, :].float())
        if idx_batch == (len(loader)-1):
            for ch in range(tsne_results.shape[1]):
                with open(f'{args.pkl_t_SNE_filename}_based_on_sub_group_size_of_{args.N_random_samples}_ch_{ch}.pkl', 'wb') as file:
                    pickle.dump(tsne[ch], file)

        batch_x = torch.from_numpy(tsne_results[0:batch_x.shape[0]]).to(device=device, dtype=torch.float32)

        output = model(batch_x)
        confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
        confidence += torch.mean(confi)
        try:
            loss = criterion(output, batch_y)
        except:
            loss = criterion(output, torch.max(batch_y, 1)[0])
        # loss = my_loss(output, batch_y)
        # cur_analog_res = [output, batch_y]
        try:
            analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
            analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()
        except:
            continue

        loss.backward()
        optimizer.step()

        y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
        y_true_all.append(batch_y.cpu().numpy())

        train_loss[idx_batch] = loss.item()
        train_loss_agg += loss.item()

        if idx_batch % 25 == 0:
            print(f'[batch_idx/total_batch_size]: {idx_batch}/{len(loader)}, with Avg Training Loss of: {train_loss_agg / (idx_batch+1)}' + \
                  f' & Avg Acc of: {metric(np.concatenate(y_true_all), np.concatenate(y_pred_all))}')

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader)
    print('\n\n')
    try:
        return np.mean(train_loss), perf, acc, confidence, analog_res
    except:
        return np.mean(train_loss), perf, acc, confidence

def _validate_subgroup_t_SNE_with_torch(model, loader, tsne_loader, criterion, device, metric, args=None):
    confidence = 0
    # validation loop
    model.eval()

    val_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []

    tsne_data = next(iter(tsne_loader))
    tsne = []
    bn = nn.BatchNorm1d(num_features=tsne_data.shape[1]).to(device=device)



    with torch.no_grad():
        for idx_batch, (batch_x, batch_y) in enumerate(loader):
            # batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.int64)

            con_data_before_norm = torch.cat((batch_x, tsne_data), dim=0).to(device=device, dtype=torch.float32)
            con_data = bn(con_data_before_norm)
            if idx_batch == 0:
                tsne_results = np.zeros((con_data.shape[0], con_data.shape[1], args.t_SNE_n_components))
                for ch in range(tsne_results.shape[1]):
                    with open(
                            f'{args.pkl_t_SNE_filename}_based_on_sub_group_size_of_{args.N_random_samples}_ch_{ch}.pkl',
                            'rb') as file:
                        tsne.append(pickle.load(file))

            for ch in range(tsne_results.shape[1]):
                tsne_results[:, ch, :] = tsne[ch].fit_transform(con_data[:, ch, :])

            batch_x = torch.from_numpy(tsne_results[0:batch_x.shape[0]]).to(device=device, dtype=torch.float32)

            output = model.forward(batch_x)

            confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
            confidence += torch.mean(confi)
            try:
                loss = criterion(output, batch_y)
            except:
                loss = criterion(output, torch.max(batch_y, 1)[0])
            # loss = my_loss(output, batch_y)
            # cur_analog_res = [output, batch_y]
            # analog_res.append(cur_analog_res)
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()
            val_loss[idx_batch] = loss.item()

            y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
            y_true_all.append(batch_y.cpu().numpy())

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader)

    try:
        return np.mean(val_loss), perf, acc, confidence, analog_res
    except:
        return np.mean(val_loss), perf, acc, confidence

def _do_train_with_esn(model, loader, optimizer, criterion, device, metric, args=None):
    confidence = 0
    # training loop
    model.train()
    train_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []
    tmp_hidden = 0
    for idx_batch, (batch_x, batch_y) in enumerate(loader):
        optimizer.zero_grad()
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.int64)
        try:
            if idx_batch > 0:
                output, hidden_tmp = model(batch_x, state='train', hidden=hidden)
            else:
                output, hidden_tmp = model(batch_x, state='train')

            if hidden_tmp.shape[1] == batch_x.shape[0]:
                hidden = hidden_tmp
            else:
                pass
            tmp_hidden += torch.mean(hidden.transpose(0, 1), dim=0, keepdim=True)
            # tmp_hidden += hidden
            confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
            confidence += torch.mean(confi)
            loss = criterion(output, batch_y)
            # loss = my_loss(output, batch_y)
            # cur_analog_res = [output, batch_y]
            analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
            analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
            y_true_all.append(batch_y.cpu().numpy())

            train_loss[idx_batch] = loss.item()
        except:
            pass

    hidden = tmp_hidden / len(loader)
    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader)
    cur_best_model = copy.deepcopy(model)
    try:
        return np.mean(train_loss), perf, acc, confidence, analog_res, hidden, cur_best_model
    except:
        return np.mean(train_loss), perf, acc, confidence, hidden, cur_best_model

def _validate_with_esn(model, loader, criterion, device, metric, args=None, hidden=None):
    confidence = 0
    # validation loop
    model.eval()

    val_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []
    with torch.no_grad():
        for idx_batch, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.int64)
            output, _ = model.forward(batch_x, state='test', hidden=hidden)

            confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
            confidence += torch.mean(confi)
            loss = criterion(output, batch_y)
            # loss = my_loss(output, batch_y)
            # cur_analog_res = [output, batch_y]
            # analog_res.append(cur_analog_res)
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), 0:args.n_classes] = (F.softmax(output, dim=1)).cpu().detach().numpy()
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), args.n_classes] = batch_y.cpu().detach().numpy()
            val_loss[idx_batch] = loss.item()

            y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
            y_true_all.append(batch_y.cpu().numpy())

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader)

    try:
        return np.mean(val_loss), perf, acc, confidence, analog_res
    except:
        return np.mean(val_loss), perf, acc, confidence



def train_kfold(model, loaders, optimizer, criterion_train, criterion_val, n_epochs,
          patience, device, metric=None, Best_model_save_path=None, args=None):
    """Training function.

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    metric : None | callable
        Metric to use to evaluate performance on the training and
        validation sets. Defaults to balanced accuracy.

    Returns
    -------
    best_model : instance of nn.Module
        The model that led to the best prediction on the validation
        dataset.
    history : list of dicts
        Training history (loss, accuracy, etc.)
    """
    best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    waiting = 0
    history = list()

    if metric is None:
        metric = balanced_accuracy_score

    print('epoch \t train_loss \t valid_loss \t train_perf \t valid_perf \t train_acc \t valid_acc \t train_confidence \t valid_confidence')
    print('--------------------------------------------------------------------------------------------------------------------------------------')
    loader_train = loaders['train_loader']
    loader_valid = loaders['val_loader']
    for epoch in range(1, n_epochs + 1):
        train_loss, train_perf, train_acc, train_confidence, analog_res_train = _do_train(
            model, loader_train, optimizer, criterion_train, device, metric=metric, args=args)
        valid_loss, valid_perf, valid_acc, valid_confidence, analog_res_val = _validate(
            model, loader_valid, criterion_val, device, metric=metric, args=args)
        history.append(
            {'epoch': epoch,
             'train_loss': train_loss, 'valid_loss': valid_loss,
             'train_perf': train_perf, 'valid_perf': valid_perf,
             'train_acc': train_acc.item(), 'valid_acc': valid_acc.item(),
             'train_confidence': train_confidence.item(), 'valid_confidence': valid_confidence.item()})

        print(f'{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} '
              f'\t {train_perf:0.4f} \t {valid_perf:0.4f}  '
              f'\t {train_acc:0.4f} \t {valid_acc:0.4f}  '
              f'\t {train_confidence:0.4f} \t {valid_confidence:0.4f}  ')

        # model saving
        if valid_loss < best_valid_loss:
            print(f'best val loss {best_valid_loss:.4f} -> {valid_loss:.4f}')
            best_valid_loss = valid_loss
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path)
            waiting = 0
        else:
            waiting += 1

        # model early stopping
        if waiting >= patience:
            print(f'Stop training at epoch {epoch}')
            print(f'Best val loss : {best_valid_loss:.4f}')
            break
    try:
        return best_model, history, analog_res_train, analog_res_val
    except:
        return best_model, history



def train(model, loaders, optimizer, criterion, n_epochs,
          patience, device, metric=None, Best_model_save_path=None, Best_model_save_path_pickle=None, args=None, best_valid_loss=np.inf, tr_subj=None, exm_subj=None):
    """Training function.

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    metric : None | callable
        Metric to use to evaluate performance on the training and
        validation sets. Defaults to balanced accuracy.

    Returns
    -------
    best_model : instance of nn.Module
        The model that led to the best prediction on the validation
        dataset.
    history : list of dicts
        Training history (loss, accuracy, etc.)
    """
    # best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    waiting = 0
    history = list()
    lr_cur = []

    if metric is None:
        metric = balanced_accuracy_score

    print('epoch \t train_loss \t valid_loss \t train_perf \t valid_perf \t train_acc \t valid_acc \t train_confidence \t valid_confidence')
    print('--------------------------------------------------------------------------------------------------------------------------------------')
    loader_train = loaders['train_loader']
    loader_valid = loaders['val_loader']
    if args.lr_decay:
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        # scheduler = ReduceLROnPlateau(optimizer, 'min')
        decayRate = 0.99
        lambda2 = lambda epoch: decayRate ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    best_val_acc = 0
    best_train_acc = 0
    for epoch in range(1, n_epochs + 1):
        train_loss, train_perf, train_acc, train_confidence, analog_res_train = _do_train(
            model, loader_train, optimizer, criterion, device, metric=metric, args=args)
        valid_loss, valid_perf, valid_acc, valid_confidence, analog_res_val = _validate(
            model, loader_valid, criterion, device, metric=metric, args=args)
        if args.lr_decay:
            scheduler.step()
            print(f'LR is: {scheduler.get_lr()[0]} @ Epoch: {epoch}')
            lr_cur.append(scheduler.get_lr()[0])
        if best_val_acc <= valid_acc:
            best_val_acc = valid_acc
            print('New val acc')
        if best_train_acc <= train_acc:
            best_train_acc = train_acc
        history.append(
            {'epoch': epoch,
             'train_loss': train_loss, 'valid_loss': valid_loss,
             'train_perf': train_perf, 'valid_perf': valid_perf,
             'train_acc': train_acc.item(), 'valid_acc': valid_acc.item(),
             'train_confidence': train_confidence.item(), 'valid_confidence': valid_confidence.item(),
             'Exm subj': exm_subj, 'Trained subj': tr_subj})

        print(f'{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} '
              f'\t {train_perf:0.4f} \t {valid_perf:0.4f}  '
              f'\t {train_acc:0.4f} \t {valid_acc:0.4f}  '
              f'\t {train_confidence:0.4f} \t {valid_confidence:0.4f}  ')

        # model saving
        if valid_loss < best_valid_loss:
            print(f'best val loss {best_valid_loss:.4f} -> {valid_loss:.4f}')
            best_valid_loss = valid_loss
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path)
                try:
                    with open(Best_model_save_path_pickle, 'wb') as f:
                        pickle.dump(pickle.dumps(model), f)
                except:
                    pass
                model = model.to(device)
            waiting = 0
        else:
            waiting += 1

        # model early stopping
        if waiting >= patience:
            print(f'Stop training at epoch {epoch}')
            print(f'Best val loss : {best_valid_loss:.4f}')
            break
    try:
        # plt.figure()
        # plt.plot(range(len(lr_cur)), lr_cur)
        # plt.show(block=False)
        return best_model, history, analog_res_train, analog_res_val, best_valid_loss, best_val_acc, best_train_acc
    except:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        return best_model, history


def train_with_val_regular_and_majority_decision(model, loaders, optimizer, criterion, n_epochs,
          patience, device, metric=None, Best_model_save_path=None, Best_model_save_path_pickle=None, args=None, best_valid_loss=np.inf, tr_subj=None, exm_subj=None, total_num_of_subjects=27, best_valid_acc=0):
    """Training function.

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    metric : None | callable
        Metric to use to evaluate performance on the training and
        validation sets. Defaults to balanced accuracy.

    Returns
    -------
    best_model : instance of nn.Module
        The model that led to the best prediction on the validation
        dataset.
    history : list of dicts
        Training history (loss, accuracy, etc.)
    """
    # best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    waiting = 0
    history = list()
    lr_cur = []

    if metric is None:
        metric = balanced_accuracy_score

    print('epoch \t train_loss \t valid_loss \t train_perf \t valid_perf_regular \t valid_perf_majority \t train_acc \t valid_acc_regular \t valid_acc_majority \t train_confidence \t valid_confidence')
    print('--------------------------------------------------------------------------------------------------------------------------------------')
    loader_train = loaders['train_loader']
    loader_valid = loaders['val_loader']
    if args.lr_decay:
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        # scheduler = ReduceLROnPlateau(optimizer, 'min')
        decayRate = 0.975
        lambda2 = lambda epoch: decayRate ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    best_val_acc_regular = 0
    best_val_acc_majority = 0
    best_train_acc = 0

    val_acc_regular = np.zeros((n_epochs, 1))
    val_acc_majority = np.zeros((n_epochs, 1))
    train_acc = np.zeros((n_epochs, 1))

    for epoch in range(1, n_epochs + 1):
        train_loss, train_perf, train_acc_result, train_confidence, analog_res_train = _do_train(
            model, loader_train, optimizer, criterion, device, metric=metric, args=args)
        valid_loss, valid_perf_regular, valid_perf_majority, valid_acc_regular, valid_acc_majority, valid_confidence, analog_res_val = _validate_with_majority_and_regular(
            model, loader_valid, criterion, device, metric=metric, args=args)
        if args.lr_decay:
            scheduler.step()
            print(f'LR is: {scheduler.get_lr()[0]} @ Epoch: {epoch}')
            lr_cur.append(scheduler.get_lr()[0])

        val_acc_regular[epoch-1] = valid_acc_regular
        val_acc_majority[epoch - 1] = valid_acc_majority
        train_acc[epoch - 1] = train_acc_result

        if best_val_acc_regular <= valid_acc_regular:
            best_val_acc_regular = valid_acc_regular
            print('New val acc regular')
        if best_val_acc_majority <= valid_acc_majority:
            best_val_acc_majority = valid_acc_majority
            print('New val acc majority')
        if best_train_acc <= train_acc_result:
            best_train_acc = train_acc_result
        history.append(
            {'epoch': epoch,
             'train_loss': train_loss, 'valid_loss': valid_loss,
             'train_perf': train_perf, 'valid_perf_regular': valid_perf_regular, 'valid_perf_majority': valid_perf_majority,
             'train_acc': train_acc_result.item(), 'valid_acc_regular': valid_acc_regular.item(), 'valid_acc_majority': valid_acc_majority.item(),
             'train_confidence': train_confidence.item(), 'valid_confidence': valid_confidence.item(),
             'Exm subj': exm_subj, 'Trained subj': tr_subj})

        print(f'{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} '
              f'\t {train_perf:0.4f} \t {valid_perf_regular:0.4f}  \t  {valid_perf_majority:0.4f}'
              f'\t {train_acc_result:0.4f} \t {valid_acc_regular:0.4f}  \t  {valid_acc_majority:0.4f}'
              f'\t {train_confidence:0.4f} \t {valid_confidence:0.4f}  ')

        # model saving
        if valid_acc_majority > best_valid_acc:
            print(f'best val ACC {best_valid_acc:.4f} -> {valid_acc_majority:.4f}')
            best_valid_acc = valid_acc_majority
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path)
                try:
                    with open(Best_model_save_path_pickle, 'wb') as f:
                        pickle.dump(pickle.dumps(model), f)
                except:
                    pass
                model = model.to(device)
            waiting = 0
        else:
            waiting += 1

        # model early stopping
        if waiting >= patience:
            print(f'Stop training at epoch {epoch}')
            print(f'Best val loss : {best_valid_loss:.4f}')
            break
    try:
        # plt.figure()
        # plt.plot(range(len(lr_cur)), lr_cur)
        # plt.show(block=False)
        return best_model, history, analog_res_train, analog_res_val, best_valid_loss, best_val_acc_regular, best_val_acc_majority, best_train_acc, val_acc_regular, val_acc_majority, train_acc
    except:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        return best_model, history

def train_with_val_regular_and_majority_decision_with_domain_adaptation(model, loaders, optimizer, criterion, n_epochs,
          patience, device, metric=None, Best_model_save_path=None, Best_model_save_path_pickle=None, args=None, best_valid_loss=np.inf, tr_subj=None, exm_subj=None, total_num_of_subjects=27, best_valid_acc=0):
    """Training function.

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    metric : None | callable
        Metric to use to evaluate performance on the training and
        validation sets. Defaults to balanced accuracy.

    Returns
    -------
    best_model : instance of nn.Module
        The model that led to the best prediction on the validation
        dataset.
    history : list of dicts
        Training history (loss, accuracy, etc.)
    """
    # best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    waiting = 0
    history = list()
    lr_cur = []

    if metric is None:
        metric = balanced_accuracy_score

    print('epoch \t train_loss \t valid_loss \t train_perf \t valid_perf_regular \t valid_perf_majority \t train_acc \t valid_acc_regular \t valid_acc_majority \t train_confidence \t valid_confidence')
    print('--------------------------------------------------------------------------------------------------------------------------------------')
    loader_train = loaders['train_loader']
    loader_valid = loaders['val_loader']
    loader_valid_train_phase = loaders['val_loader_train_phase']
    if args.lr_decay:
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        # scheduler = ReduceLROnPlateau(optimizer, 'min')
        decayRate = 0.975
        lambda2 = lambda epoch: decayRate ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    best_val_acc_regular = 0
    best_val_acc_majority = 0
    best_train_acc = 0

    val_acc_regular = np.zeros((n_epochs, 1))
    val_acc_majority = np.zeros((n_epochs, 1))
    train_acc = np.zeros((n_epochs, 1))

    for epoch in range(1, n_epochs + 1):
        train_loss, train_perf, train_acc_result, train_confidence, analog_res_train = _do_train_domain_adaptation(
            model, loader_train, loader_valid_train_phase, optimizer, criterion, device, metric=metric, args=args)
        valid_loss, valid_perf_regular, valid_perf_majority, valid_acc_regular, valid_acc_majority, valid_confidence, analog_res_val = _validate_with_majority_and_regular(
            model, loader_valid, criterion, device, metric=metric, args=args)
        if args.lr_decay:
            scheduler.step()
            print(f'LR is: {scheduler.get_lr()[0]} @ Epoch: {epoch}')
            lr_cur.append(scheduler.get_lr()[0])

        val_acc_regular[epoch-1] = valid_acc_regular
        val_acc_majority[epoch - 1] = valid_acc_majority
        train_acc[epoch - 1] = train_acc_result

        if best_val_acc_regular <= valid_acc_regular:
            best_val_acc_regular = valid_acc_regular
            print('New val acc regular')
        if best_val_acc_majority <= valid_acc_majority:
            best_val_acc_majority = valid_acc_majority
            print('New val acc majority')
        if best_train_acc <= train_acc_result:
            best_train_acc = train_acc_result
        history.append(
            {'epoch': epoch,
             'train_loss': train_loss, 'valid_loss': valid_loss,
             'train_perf': train_perf, 'valid_perf_regular': valid_perf_regular, 'valid_perf_majority': valid_perf_majority,
             'train_acc': train_acc_result.item(), 'valid_acc_regular': valid_acc_regular.item(), 'valid_acc_majority': valid_acc_majority.item(),
             'train_confidence': train_confidence.item(), 'valid_confidence': valid_confidence.item(),
             'Exm subj': exm_subj, 'Trained subj': tr_subj})

        print(f'{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} '
              f'\t {train_perf:0.4f} \t {valid_perf_regular:0.4f}  \t  {valid_perf_majority:0.4f}'
              f'\t {train_acc_result:0.4f} \t {valid_acc_regular:0.4f}  \t  {valid_acc_majority:0.4f}'
              f'\t {train_confidence:0.4f} \t {valid_confidence:0.4f}  ')

        # model saving
        if valid_acc_majority > best_valid_acc:
            print(f'best val ACC {best_valid_acc:.4f} -> {valid_acc_majority:.4f}')
            best_valid_acc = valid_acc_majority
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path)
                try:
                    with open(Best_model_save_path_pickle, 'wb') as f:
                        pickle.dump(pickle.dumps(model), f)
                except:
                    pass
                model = model.to(device)
            waiting = 0
        else:
            waiting += 1

        # model early stopping
        if waiting >= patience:
            print(f'Stop training at epoch {epoch}')
            print(f'Best val loss : {best_valid_loss:.4f}')
            break
    try:
        # plt.figure()
        # plt.plot(range(len(lr_cur)), lr_cur)
        # plt.show(block=False)
        return best_model, history, analog_res_train, analog_res_val, best_valid_loss, best_val_acc_regular, best_val_acc_majority, best_train_acc, val_acc_regular, val_acc_majority, train_acc
    except:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        return best_model, history

def train_with_val_regular_and_majority_decision_with_domain_adaptation_MS_MDA(model, loaders, optimizer, criterion, n_epochs,
          patience, device, metric=None, Best_model_save_path=None, Best_model_save_path_pickle=None, args=None, best_valid_loss=np.inf, tr_subj=None, exm_subj=None, total_num_of_subjects=27, best_valid_acc=0):
    """Training function.

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    metric : None | callable
        Metric to use to evaluate performance on the training and
        validation sets. Defaults to balanced accuracy.

    Returns
    -------
    best_model : instance of nn.Module
        The model that led to the best prediction on the validation
        dataset.
    history : list of dicts
        Training history (loss, accuracy, etc.)
    """
    # best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    waiting = 0
    history = list()
    lr_cur = []

    if metric is None:
        metric = balanced_accuracy_score

    print('epoch \t train_loss \t valid_loss \t train_perf \t valid_perf_regular \t valid_perf_majority \t train_acc \t valid_acc_regular \t valid_acc_majority \t train_confidence \t valid_confidence')
    print('--------------------------------------------------------------------------------------------------------------------------------------')
    loader_train = loaders['train_loader']
    loader_valid = loaders['val_loader']
    loader_valid_train_phase = loaders['val_loader_train_phase']
    if args.lr_decay:
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        # scheduler = ReduceLROnPlateau(optimizer, 'min')
        decayRate = 0.975
        lambda2 = lambda epoch: decayRate ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    best_val_acc_regular = 0
    best_val_acc_majority = 0
    best_train_acc = 0

    val_acc_regular = np.zeros((n_epochs, 1))
    val_acc_majority = np.zeros((n_epochs, 1))
    train_acc = np.zeros((n_epochs, 1))

    for epoch in range(1, n_epochs + 1):
        train_loss, train_perf, train_acc_result, train_confidence, analog_res_train = _do_train_domain_adaptation(
            model, loader_train, loader_valid_train_phase, optimizer, criterion, device, metric=metric, args=args)
        valid_loss, valid_perf_regular, valid_perf_majority, valid_acc_regular, valid_acc_majority, valid_confidence, analog_res_val = _validate_with_majority_and_regular(
            model, loader_valid, criterion, device, metric=metric, args=args)
        if args.lr_decay:
            scheduler.step()
            print(f'LR is: {scheduler.get_lr()[0]} @ Epoch: {epoch}')
            lr_cur.append(scheduler.get_lr()[0])

        val_acc_regular[epoch-1] = valid_acc_regular
        val_acc_majority[epoch - 1] = valid_acc_majority
        train_acc[epoch - 1] = train_acc_result

        if best_val_acc_regular <= valid_acc_regular:
            best_val_acc_regular = valid_acc_regular
            print('New val acc regular')
        if best_val_acc_majority <= valid_acc_majority:
            best_val_acc_majority = valid_acc_majority
            print('New val acc majority')
        if best_train_acc <= train_acc_result:
            best_train_acc = train_acc_result
        history.append(
            {'epoch': epoch,
             'train_loss': train_loss, 'valid_loss': valid_loss,
             'train_perf': train_perf, 'valid_perf_regular': valid_perf_regular, 'valid_perf_majority': valid_perf_majority,
             'train_acc': train_acc_result.item(), 'valid_acc_regular': valid_acc_regular.item(), 'valid_acc_majority': valid_acc_majority.item(),
             'train_confidence': train_confidence.item(), 'valid_confidence': valid_confidence.item(),
             'Exm subj': exm_subj, 'Trained subj': tr_subj})

        print(f'{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} '
              f'\t {train_perf:0.4f} \t {valid_perf_regular:0.4f}  \t  {valid_perf_majority:0.4f}'
              f'\t {train_acc_result:0.4f} \t {valid_acc_regular:0.4f}  \t  {valid_acc_majority:0.4f}'
              f'\t {train_confidence:0.4f} \t {valid_confidence:0.4f}  ')

        # model saving
        if valid_acc_majority > best_valid_acc:
            print(f'best val ACC {best_valid_acc:.4f} -> {valid_acc_majority:.4f}')
            best_valid_acc = valid_acc_majority
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path)
                try:
                    with open(Best_model_save_path_pickle, 'wb') as f:
                        pickle.dump(pickle.dumps(model), f)
                except:
                    pass
                model = model.to(device)
            waiting = 0
        else:
            waiting += 1

        # model early stopping
        if waiting >= patience:
            print(f'Stop training at epoch {epoch}')
            print(f'Best val loss : {best_valid_loss:.4f}')
            break
    try:
        # plt.figure()
        # plt.plot(range(len(lr_cur)), lr_cur)
        # plt.show(block=False)
        return best_model, history, analog_res_train, analog_res_val, best_valid_loss, best_val_acc_regular, best_val_acc_majority, best_train_acc, val_acc_regular, val_acc_majority, train_acc
    except:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        return best_model, history

def train_with_val_regular_and_majority_decision_fusion_EEGNet(model, loaders, optimizer, criterion, n_epochs,
          patience, device, metric=None, Best_model_save_path=None, Best_model_save_path_pickle=None, args=None, best_valid_loss=np.inf, tr_subj=None, exm_subj=None, total_num_of_subjects=27, best_valid_acc=0):
    """Training function.

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    metric : None | callable
        Metric to use to evaluate performance on the training and
        validation sets. Defaults to balanced accuracy.

    Returns
    -------
    best_model : instance of nn.Module
        The model that led to the best prediction on the validation
        dataset.
    history : list of dicts
        Training history (loss, accuracy, etc.)
    """
    # best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    waiting = 0
    history = list()
    lr_cur = []

    if metric is None:
        metric = balanced_accuracy_score

    # print('epoch \t train_loss \t valid_loss \t train_perf \t valid_perf_regular \t valid_perf_majority \t train_acc \t valid_acc_regular \t valid_acc_majority \t train_confidence \t valid_confidence')
    # print('--------------------------------------------------------------------------------------------------------------------------------------')
    loader_train = loaders['train_loader']
    loader_valid = loaders['val_loader']
    if args.lr_decay:
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        # scheduler = ReduceLROnPlateau(optimizer, 'min')
        decayRate = 0.975
        lambda2 = lambda epoch: decayRate ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    best_val_acc_regular = 0
    best_val_acc_majority = 0
    best_train_acc = 0

    val_acc_regular = np.zeros((n_epochs, 1))
    val_acc_majority = np.zeros((n_epochs, 1))
    train_acc = np.zeros((n_epochs, 1))

    for epoch in range(1, n_epochs + 1):
        train_loss, train_perf, train_acc_result, train_confidence, analog_res_train = _do_train_fusion_EEGNet(
            model, loader_train, optimizer, criterion, device, metric=metric, args=args)
        valid_loss, valid_perf_regular, valid_perf_majority, valid_acc_regular, valid_acc_majority, valid_confidence, analog_res_val = _validate_with_majority_and_regular_fusion_EEGNet(
            model, loader_valid, criterion, device, metric=metric, args=args)
        if args.lr_decay:
            scheduler.step()
            print(f'LR is: {scheduler.get_lr()[0]} @ Epoch: {epoch}')
            lr_cur.append(scheduler.get_lr()[0])

        val_acc_regular[epoch-1] = valid_acc_regular
        val_acc_majority[epoch - 1] = valid_acc_majority
        train_acc[epoch - 1] = train_acc_result

        if best_val_acc_regular <= valid_acc_regular:
            best_val_acc_regular = valid_acc_regular
            print('New val acc regular')
        if best_val_acc_majority <= valid_acc_majority:
            best_val_acc_majority = valid_acc_majority
            print('New val acc majority')
        if best_train_acc <= train_acc_result:
            best_train_acc = train_acc_result
        history.append(
            {'epoch': epoch,
             'train_loss': train_loss, 'valid_loss': valid_loss,
             'train_perf': train_perf, 'valid_perf_regular': valid_perf_regular, 'valid_perf_majority': valid_perf_majority,
             'train_acc': train_acc_result.item(), 'valid_acc_regular': valid_acc_regular.item(), 'valid_acc_majority': valid_acc_majority.item(),
             'train_confidence': train_confidence.item(), 'valid_confidence': valid_confidence.item(),
             'Exm subj': exm_subj, 'Trained subj': tr_subj})

        # print(f'{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} '
        #       f'\t {train_perf:0.4f} \t {valid_perf_regular:0.4f}  \t  {valid_perf_majority:0.4f}'
        #       f'\t {train_acc_result:0.4f} \t {valid_acc_regular:0.4f}  \t  {valid_acc_majority:0.4f}'
        #       f'\t {train_confidence:0.4f} \t {valid_confidence:0.4f}  ')
        # Validation accuracy
        params = ["acc", "auc", "fmeasure"]
        print(params)
        print("Train - ", evaluate_eeg(model, loader_train.dataset.X, loader_train.dataset.y, params))
        print("Validation / Test - ", evaluate_eeg(model, loader_valid.dataset.X, loader_valid.dataset.y, params))
        # evaluate_eeg(model, )

        # model saving
        if valid_acc_regular > best_valid_acc:
            print(f'best val ACC {best_valid_acc:.4f} -> {valid_acc_regular:.4f}')
            best_valid_acc = valid_acc_regular
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path)
                try:
                    with open(Best_model_save_path_pickle, 'wb') as f:
                        pickle.dump(pickle.dumps(model), f)
                except:
                    pass
                model = model.to(device)
            waiting = 0
        else:
            waiting += 1

        # model early stopping
        if waiting >= patience:
            print(f'Stop training at epoch {epoch}')
            print(f'Best val loss : {best_valid_loss:.4f}')
            break
    try:
        # plt.figure()
        # plt.plot(range(len(lr_cur)), lr_cur)
        # plt.show(block=False)
        return best_model, history, analog_res_train, analog_res_val, best_valid_loss, best_val_acc_regular, best_val_acc_majority, best_train_acc, val_acc_regular, val_acc_majority, train_acc
    except:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        return best_model, history

def train_with_val_regular_and_majority_decision_fusion_big_DS(model, loaders, optimizer, criterion, n_epochs,
          patience, device, metric=None, Best_model_save_path=None, Best_model_save_path_pickle=None, args=None, best_valid_loss=np.inf, tr_subj=None, exm_subj=None, total_num_of_subjects=27, best_valid_acc=0):
    """Training function.

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    metric : None | callable
        Metric to use to evaluate performance on the training and
        validation sets. Defaults to balanced accuracy.

    Returns
    -------
    best_model : instance of nn.Module
        The model that led to the best prediction on the validation
        dataset.
    history : list of dicts
        Training history (loss, accuracy, etc.)
    """
    # best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    waiting = 0
    history = list()
    lr_cur = []

    if metric is None:
        metric = balanced_accuracy_score

    print('epoch \t train_loss \t valid_loss \t train_perf \t valid_perf_regular \t valid_perf_majority \t train_acc \t valid_acc_regular \t valid_acc_majority \t train_confidence \t valid_confidence')
    print('--------------------------------------------------------------------------------------------------------------------------------------')
    loader_train = loaders['train_loader']
    loader_valid = loaders['val_loader']
    if args.lr_decay:
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        # scheduler = ReduceLROnPlateau(optimizer, 'min')
        decayRate = 0.975
        lambda2 = lambda epoch: decayRate ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    best_val_acc_regular = 0
    best_val_acc_majority = 0
    best_train_acc = 0

    val_acc_regular = np.zeros((n_epochs, 1))
    val_acc_majority = np.zeros((n_epochs, 1))
    train_acc = np.zeros((n_epochs, 1))

    for epoch in range(1, n_epochs + 1):
        train_loss, train_perf, train_acc_result, train_confidence, analog_res_train = _do_train_fusion_big_DS(
            model, loader_train, optimizer, criterion, device, metric=metric, args=args)
        valid_loss, valid_perf_regular, valid_perf_majority, valid_acc_regular, valid_acc_majority, valid_confidence, analog_res_val = _validate_with_majority_and_regular_fusion_big_DS(
            model, loader_valid, criterion, device, metric=metric, args=args)
        if args.lr_decay:
            scheduler.step()
            print(f'LR is: {scheduler.get_lr()[0]} @ Epoch: {epoch}')
            lr_cur.append(scheduler.get_lr()[0])

        val_acc_regular[epoch-1] = valid_acc_regular
        val_acc_majority[epoch - 1] = valid_acc_majority
        train_acc[epoch - 1] = train_acc_result

        if best_val_acc_regular <= valid_acc_regular:
            best_val_acc_regular = valid_acc_regular
            print('New val acc regular')
        if best_val_acc_majority <= valid_acc_majority:
            best_val_acc_majority = valid_acc_majority
            print('New val acc majority')
        if best_train_acc <= train_acc_result:
            best_train_acc = train_acc_result
        history.append(
            {'epoch': epoch,
             'train_loss': train_loss, 'valid_loss': valid_loss,
             'train_perf': train_perf, 'valid_perf_regular': valid_perf_regular, 'valid_perf_majority': valid_perf_majority,
             'train_acc': train_acc_result.item(), 'valid_acc_regular': valid_acc_regular.item(), 'valid_acc_majority': valid_acc_majority.item(),
             'train_confidence': train_confidence.item(), 'valid_confidence': valid_confidence.item(),
             'Exm subj': exm_subj, 'Trained subj': tr_subj})

        print(f'{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} '
              f'\t {train_perf:0.4f} \t {valid_perf_regular:0.4f}  \t  {valid_perf_majority:0.4f}'
              f'\t {train_acc_result:0.4f} \t {valid_acc_regular:0.4f}  \t  {valid_acc_majority:0.4f}'
              f'\t {train_confidence:0.4f} \t {valid_confidence:0.4f}  ')

        # model saving
        if valid_acc_regular > best_valid_acc:
            print(f'best val ACC {best_valid_acc:.4f} -> {valid_acc_regular:.4f}')
            best_valid_acc = valid_acc_regular
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path)
                try:
                    with open(Best_model_save_path_pickle, 'wb') as f:
                        pickle.dump(pickle.dumps(model), f)
                except:
                    pass
                model = model.to(device)
            waiting = 0
        else:
            waiting += 1

        # model early stopping
        if waiting >= patience:
            print(f'Stop training at epoch {epoch}')
            print(f'Best val loss : {best_valid_loss:.4f}')
            break
    try:
        # plt.figure()
        # plt.plot(range(len(lr_cur)), lr_cur)
        # plt.show(block=False)
        return best_model, history, analog_res_train, analog_res_val, best_valid_loss, best_val_acc_regular, best_val_acc_majority, best_train_acc, val_acc_regular, val_acc_majority, train_acc
    except:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        return best_model, history

def train_with_val_regular_and_majority_decision_fusion_our_target_DS(model, loaders, optimizer, criterion, n_epochs,
          patience, device, metric=None, Best_model_save_path=None, Best_model_save_path_pickle=None, args=None, best_valid_loss=np.inf, tr_subj=None, exm_subj=None, total_num_of_subjects=27, best_valid_acc=0):
    """Training function.

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    metric : None | callable
        Metric to use to evaluate performance on the training and
        validation sets. Defaults to balanced accuracy.

    Returns
    -------
    best_model : instance of nn.Module
        The model that led to the best prediction on the validation
        dataset.
    history : list of dicts
        Training history (loss, accuracy, etc.)
    """
    # best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    waiting = 0
    history = list()
    lr_cur = []

    if metric is None:
        metric = balanced_accuracy_score

    print('epoch \t train_loss \t valid_loss \t train_perf \t valid_perf_regular \t valid_perf_majority \t train_acc \t valid_acc_regular \t valid_acc_majority \t train_confidence \t valid_confidence')
    print('--------------------------------------------------------------------------------------------------------------------------------------')
    loader_train = loaders['train_loader']
    loader_valid = loaders['val_loader']
    if args.lr_decay:
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        # scheduler = ReduceLROnPlateau(optimizer, 'min')
        decayRate = 0.975
        lambda2 = lambda epoch: decayRate ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    best_val_acc_regular = 0
    best_val_acc_majority = 0
    best_train_acc = 0

    val_acc_regular = np.zeros((n_epochs, 1))
    val_acc_majority = np.zeros((n_epochs, 1))
    train_acc = np.zeros((n_epochs, 1))

    for epoch in range(1, n_epochs + 1):
        train_loss, train_perf, train_acc_result, train_confidence, analog_res_train = _do_train_fusion_our_target_DS(
            model, loader_train, optimizer, criterion, device, metric=metric, args=args)
        valid_loss, valid_perf_regular, valid_perf_majority, valid_acc_regular, valid_acc_majority, valid_confidence, analog_res_val = _validate_with_majority_and_regular_fusion_our_target_DS(
            model, loader_valid, criterion, device, metric=metric, args=args)
        if args.lr_decay:
            scheduler.step()
            print(f'LR is: {scheduler.get_lr()[0]} @ Epoch: {epoch}')
            lr_cur.append(scheduler.get_lr()[0])

        val_acc_regular[epoch-1] = valid_acc_regular
        val_acc_majority[epoch - 1] = valid_acc_majority
        train_acc[epoch - 1] = train_acc_result

        if best_val_acc_regular <= valid_acc_regular:
            best_val_acc_regular = valid_acc_regular
            print('New val acc regular')
        if best_val_acc_majority <= valid_acc_majority:
            best_val_acc_majority = valid_acc_majority
            print('New val acc majority')
        if best_train_acc <= train_acc_result:
            best_train_acc = train_acc_result
        history.append(
            {'epoch': epoch,
             'train_loss': train_loss, 'valid_loss': valid_loss,
             'train_perf': train_perf, 'valid_perf_regular': valid_perf_regular, 'valid_perf_majority': valid_perf_majority,
             'train_acc': train_acc_result.item(), 'valid_acc_regular': valid_acc_regular.item(), 'valid_acc_majority': valid_acc_majority.item(),
             'train_confidence': train_confidence.item(), 'valid_confidence': valid_confidence.item(),
             'Exm subj': exm_subj, 'Trained subj': tr_subj})

        print(f'{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} '
              f'\t {train_perf:0.4f} \t {valid_perf_regular:0.4f}  \t  {valid_perf_majority:0.4f}'
              f'\t {train_acc_result:0.4f} \t {valid_acc_regular:0.4f}  \t  {valid_acc_majority:0.4f}'
              f'\t {train_confidence:0.4f} \t {valid_confidence:0.4f}  ')

        # model saving
        if valid_acc_majority > best_valid_acc:
            print(f'best val ACC {best_valid_acc:.4f} -> {valid_acc_majority:.4f}')
            best_valid_acc = valid_acc_majority
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path)
                try:
                    with open(Best_model_save_path_pickle, 'wb') as f:
                        pickle.dump(pickle.dumps(model), f)
                except:
                    pass
                model = model.to(device)
            waiting = 0
        else:
            waiting += 1

        # model early stopping
        if waiting >= patience:
            print(f'Stop training at epoch {epoch}')
            print(f'Best val loss : {best_valid_loss:.4f}')
            break
    try:
        # plt.figure()
        # plt.plot(range(len(lr_cur)), lr_cur)
        # plt.show(block=False)
        return best_model, history, analog_res_train, analog_res_val, best_valid_loss, best_val_acc_regular, best_val_acc_majority, best_train_acc, val_acc_regular, val_acc_majority, train_acc
    except:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        return best_model, history

def train_with_val_regular_and_majority_decision_without_transform(model, loaders, optimizer, criterion, n_epochs,
          patience, device, metric=None, Best_model_save_path=None, Best_model_save_path_pickle=None, args=None, best_valid_loss=np.inf, tr_subj=None, exm_subj=None, total_num_of_subjects=27, best_valid_acc=0, tot_best_valid_acc=0):
    """Training function.

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    metric : None | callable
        Metric to use to evaluate performance on the training and
        validation sets. Defaults to balanced accuracy.

    Returns
    -------
    best_model : instance of nn.Module
        The model that led to the best prediction on the validation
        dataset.
    history : list of dicts
        Training history (loss, accuracy, etc.)
    """
    # best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    waiting = 0
    history = list()
    lr_cur = []

    if metric is None:
        metric = balanced_accuracy_score

    print('epoch \t train_loss \t valid_loss \t train_perf \t valid_perf_regular \t valid_perf_majority \t train_acc \t valid_acc_regular \t valid_acc_majority \t train_confidence \t valid_confidence')
    print('--------------------------------------------------------------------------------------------------------------------------------------')
    loader_train = loaders['train_loader']
    loader_valid = loaders['val_loader']
    if args.lr_decay:
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        # scheduler = ReduceLROnPlateau(optimizer, 'min')
        decayRate = 0.975
        lambda2 = lambda epoch: decayRate ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    best_val_acc_regular = 0
    best_val_acc_majority = 0
    best_train_acc = 0

    val_acc_regular = np.zeros((n_epochs, 1))
    val_acc_majority = np.zeros((n_epochs, 1))
    train_acc = np.zeros((n_epochs, 1))

    for epoch in range(1, n_epochs + 1):
        train_loss, train_perf, train_acc_result, train_confidence, analog_res_train = _do_train_without_transform(
            model, loader_train, optimizer, criterion, device, metric=metric, args=args)
        valid_loss, valid_perf_regular, valid_perf_majority, valid_acc_regular, valid_acc_majority, valid_confidence, analog_res_val = _validate_with_majority_and_regular_without_transform(
            model, loader_valid, criterion, device, metric=metric, args=args)
        if args.lr_decay:
            scheduler.step()
            print(f'LR is: {scheduler.get_last_lr()} @ Epoch: {epoch}')
            lr_cur.append(scheduler.get_last_lr())

        val_acc_regular[epoch-1] = valid_acc_regular
        val_acc_majority[epoch - 1] = valid_acc_majority
        train_acc[epoch - 1] = train_acc_result

        if best_val_acc_regular <= valid_acc_regular:
            best_val_acc_regular = valid_acc_regular
            print('New val acc regular')
        if best_val_acc_majority <= valid_acc_majority:
            best_val_acc_majority = valid_acc_majority
            print('New val acc majority')
        if best_train_acc <= train_acc_result:
            best_train_acc = train_acc_result
        history.append(
            {'epoch': epoch,
             'train_loss': train_loss, 'valid_loss': valid_loss,
             'train_perf': train_perf, 'valid_perf_regular': valid_perf_regular, 'valid_perf_majority': valid_perf_majority,
             'train_acc': train_acc_result.item(), 'valid_acc_regular': valid_acc_regular.item(), 'valid_acc_majority': valid_acc_majority.item(),
             'train_confidence': train_confidence.item(), 'valid_confidence': valid_confidence.item(),
             'Exm subj': exm_subj, 'Trained subj': tr_subj})

        print(f'{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} '
              f'\t {train_perf:0.4f} \t {valid_perf_regular:0.4f}  \t  {valid_perf_majority:0.4f}'
              f'\t {train_acc_result:0.4f} \t {valid_acc_regular:0.4f}  \t  {valid_acc_majority:0.4f}'
              f'\t {train_confidence:0.4f} \t {valid_confidence:0.4f}  ')

        # model saving
        if valid_acc_majority > best_valid_acc:
            print(f'best val ACC {best_valid_acc:.4f} -> {valid_acc_majority:.4f}')
            best_valid_acc = valid_acc_majority
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path + f'_in_subject_#{exm_subj}.pt')
                try:
                    with open(Best_model_save_path_pickle + f'_in_subject_#{exm_subj}.pik', 'wb') as f:
                        pickle.dump(pickle.dumps(model), f)
                except:
                    pass
                model = model.to(device)
            waiting = 0
        else:
            waiting += 1

        # model saving
        if valid_acc_majority > tot_best_valid_acc:
            print(f'Total best val ACC {tot_best_valid_acc:.4f} -> {valid_acc_majority:.4f}')
            tot_best_valid_acc = valid_acc_majority
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path + f'_in_subject_#{exm_subj}_best_total.pt')
                try:
                    with open(Best_model_save_path_pickle + f'_in_subject_#{exm_subj}_best_total.pik', 'wb') as f:
                        pickle.dump(pickle.dumps(model), f)
                except:
                    pass
                model = model.to(device)
            waiting = 0
        else:
            waiting += 1

        # model early stopping
        if waiting >= patience:
            print(f'Stop training at epoch {epoch}')
            print(f'Best val loss : {best_valid_loss:.4f}')
            break
    try:
        # plt.figure()
        # plt.plot(range(len(lr_cur)), lr_cur)
        # plt.show(block=False)
        return best_model, history, analog_res_train, analog_res_val, best_valid_loss, best_val_acc_regular, best_val_acc_majority, best_train_acc, val_acc_regular, val_acc_majority, train_acc
    except:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        return best_model, history

def train_with_val_regular_and_majority_decision_without_transform_with_unsupervised(model, loaders, optimizer, criterion, n_epochs,
          patience, device, metric=None, Best_model_save_path=None, Best_model_save_path_pickle=None, args=None, best_valid_loss=np.inf, tr_subj=None, exm_subj=None, total_num_of_subjects=27, best_valid_acc=0, tot_best_valid_acc=0):
    """Training function.

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    metric : None | callable
        Metric to use to evaluate performance on the training and
        validation sets. Defaults to balanced accuracy.

    Returns
    -------
    best_model : instance of nn.Module
        The model that led to the best prediction on the validation
        dataset.
    history : list of dicts
        Training history (loss, accuracy, etc.)
    """
    # best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    waiting = 0
    history = list()
    lr_cur = []

    if metric is None:
        metric = balanced_accuracy_score

    print('epoch \t train_loss \t valid_loss \t train_perf \t valid_perf_regular \t valid_perf_majority \t train_acc \t valid_acc_regular \t valid_acc_majority \t train_confidence \t valid_confidence')
    print('--------------------------------------------------------------------------------------------------------------------------------------')
    loader_train = loaders['train_loader']
    loader_valid = loaders['val_loader']
    if args.lr_decay:
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        # scheduler = ReduceLROnPlateau(optimizer, 'min')
        decayRate = 0.975
        lambda2 = lambda epoch: decayRate ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    best_val_acc_regular = 0
    best_val_acc_majority = 0
    best_train_acc = 0

    val_acc_regular = np.zeros((n_epochs, 1))
    val_acc_majority = np.zeros((n_epochs, 1))
    train_acc = np.zeros((n_epochs, 1))

    # Unsupervised codes
    loader = 1

    for epoch in range(1, n_epochs + 1):
        train_loss, train_perf, train_acc_result, train_confidence, analog_res_train = _do_train_without_transform(
            model, loader_train, optimizer, criterion, device, metric=metric, args=args)
        valid_loss, valid_perf_regular, valid_perf_majority, valid_acc_regular, valid_acc_majority, valid_confidence, analog_res_val = _validate_with_majority_and_regular_without_transform(
            model, loader_valid, criterion, device, metric=metric, args=args)
        if args.lr_decay:
            scheduler.step()
            print(f'LR is: {scheduler.get_last_lr()} @ Epoch: {epoch}')
            lr_cur.append(scheduler.get_last_lr())

        val_acc_regular[epoch-1] = valid_acc_regular
        val_acc_majority[epoch - 1] = valid_acc_majority
        train_acc[epoch - 1] = train_acc_result

        if best_val_acc_regular <= valid_acc_regular:
            best_val_acc_regular = valid_acc_regular
            print('New val acc regular')
        if best_val_acc_majority <= valid_acc_majority:
            best_val_acc_majority = valid_acc_majority
            print('New val acc majority')
        if best_train_acc <= train_acc_result:
            best_train_acc = train_acc_result
        history.append(
            {'epoch': epoch,
             'train_loss': train_loss, 'valid_loss': valid_loss,
             'train_perf': train_perf, 'valid_perf_regular': valid_perf_regular, 'valid_perf_majority': valid_perf_majority,
             'train_acc': train_acc_result.item(), 'valid_acc_regular': valid_acc_regular.item(), 'valid_acc_majority': valid_acc_majority.item(),
             'train_confidence': train_confidence.item(), 'valid_confidence': valid_confidence.item(),
             'Exm subj': exm_subj, 'Trained subj': tr_subj})

        print(f'{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} '
              f'\t {train_perf:0.4f} \t {valid_perf_regular:0.4f}  \t  {valid_perf_majority:0.4f}'
              f'\t {train_acc_result:0.4f} \t {valid_acc_regular:0.4f}  \t  {valid_acc_majority:0.4f}'
              f'\t {train_confidence:0.4f} \t {valid_confidence:0.4f}  ')

        # model saving
        if valid_acc_majority > best_valid_acc:
            print(f'best val ACC {best_valid_acc:.4f} -> {valid_acc_majority:.4f}')
            best_valid_acc = valid_acc_majority
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path + f'_in_subject_#{exm_subj}.pt')
                try:
                    with open(Best_model_save_path_pickle + f'_in_subject_#{exm_subj}.pik', 'wb') as f:
                        pickle.dump(pickle.dumps(model), f)
                except:
                    pass
                model = model.to(device)
            waiting = 0
        else:
            waiting += 1

        # model saving
        if valid_acc_majority > tot_best_valid_acc:
            print(f'Total best val ACC {tot_best_valid_acc:.4f} -> {valid_acc_majority:.4f}')
            tot_best_valid_acc = valid_acc_majority
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path + f'_in_subject_#{exm_subj}_best_total.pt')
                try:
                    with open(Best_model_save_path_pickle + f'_in_subject_#{exm_subj}_best_total.pik', 'wb') as f:
                        pickle.dump(pickle.dumps(model), f)
                except:
                    pass
                model = model.to(device)
            waiting = 0
        else:
            waiting += 1

        # model early stopping
        if waiting >= patience:
            print(f'Stop training at epoch {epoch}')
            print(f'Best val loss : {best_valid_loss:.4f}')
            break
    try:
        # plt.figure()
        # plt.plot(range(len(lr_cur)), lr_cur)
        # plt.show(block=False)
        return best_model, history, analog_res_train, analog_res_val, best_valid_loss, best_val_acc_regular, best_val_acc_majority, best_train_acc, val_acc_regular, val_acc_majority, train_acc
    except:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        return best_model, history

def train_with_val_regular_and_majority_decision_without_transform_with_unsupervised_our_loss(model, loaders, optimizer, criterion, n_epochs,
          patience, device, metric=None, Best_model_save_path=None, Best_model_save_path_pickle=None, args=None, best_valid_loss=np.inf, tr_subj=None, exm_subj=None, total_num_of_subjects=27, best_valid_acc=0, tot_best_valid_acc=0):
    """Training function.

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    metric : None | callable
        Metric to use to evaluate performance on the training and
        validation sets. Defaults to balanced accuracy.

    Returns
    -------
    best_model : instance of nn.Module
        The model that led to the best prediction on the validation
        dataset.
    history : list of dicts
        Training history (loss, accuracy, etc.)
    """
    # best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    waiting = 0
    history = list()
    lr_cur = []

    if metric is None:
        metric = balanced_accuracy_score

    print('epoch \t train_loss \t valid_loss \t train_perf \t valid_perf_regular \t valid_perf_majority \t train_acc \t valid_acc_regular \t valid_acc_majority \t train_confidence \t valid_confidence')
    print('--------------------------------------------------------------------------------------------------------------------------------------')
    loader_train = loaders['train_loader']
    loader_valid = loaders['val_loader']
    if args.lr_decay:
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        # scheduler = ReduceLROnPlateau(optimizer, 'min')
        decayRate = 0.975
        lambda2 = lambda epoch: decayRate ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    best_val_acc_regular = 0
    best_val_acc_majority = 0
    best_train_acc = 0

    val_acc_regular = np.zeros((n_epochs, 1))
    val_acc_majority = np.zeros((n_epochs, 1))
    train_acc = np.zeros((n_epochs, 1))

    # Unsupervised codes
    loader = 1

    for epoch in range(1, n_epochs + 1):
        train_loss, train_perf, train_acc_result, train_confidence, analog_res_train = _do_train_without_transform_with_our_loss(
            model, loader_train, optimizer, criterion, device, metric=metric, args=args)
        valid_loss, valid_perf_regular, valid_perf_majority, valid_acc_regular, valid_acc_majority, valid_confidence, analog_res_val = _validate_with_majority_and_regular_without_transform_with_our_loss(
            model, loader_valid, criterion, device, metric=metric, args=args)
        if args.lr_decay:
            scheduler.step()
            print(f'LR is: {scheduler.get_last_lr()} @ Epoch: {epoch}')
            lr_cur.append(scheduler.get_last_lr())

        val_acc_regular[epoch-1] = valid_acc_regular
        val_acc_majority[epoch - 1] = valid_acc_majority
        train_acc[epoch - 1] = train_acc_result

        if best_val_acc_regular <= valid_acc_regular:
            best_val_acc_regular = valid_acc_regular
            print('New val acc regular')
        if best_val_acc_majority <= valid_acc_majority:
            best_val_acc_majority = valid_acc_majority
            print('New val acc majority')
        if best_train_acc <= train_acc_result:
            best_train_acc = train_acc_result
        history.append(
            {'epoch': epoch,
             'train_loss': train_loss, 'valid_loss': valid_loss,
             'train_perf': train_perf, 'valid_perf_regular': valid_perf_regular, 'valid_perf_majority': valid_perf_majority,
             'train_acc': train_acc_result.item(), 'valid_acc_regular': valid_acc_regular.item(), 'valid_acc_majority': valid_acc_majority.item(),
             'train_confidence': train_confidence.item(), 'valid_confidence': valid_confidence.item(),
             'Exm subj': exm_subj, 'Trained subj': tr_subj})

        print(f'{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} '
              f'\t {train_perf:0.4f} \t {valid_perf_regular:0.4f}  \t  {valid_perf_majority:0.4f}'
              f'\t {train_acc_result:0.4f} \t {valid_acc_regular:0.4f}  \t  {valid_acc_majority:0.4f}'
              f'\t {train_confidence:0.4f} \t {valid_confidence:0.4f}  ')

        # model saving
        if valid_acc_majority > best_valid_acc:
            print(f'best val ACC {best_valid_acc:.4f} -> {valid_acc_majority:.4f}')
            best_valid_acc = valid_acc_majority
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path + f'_in_subject_#{exm_subj}.pt')
                try:
                    with open(Best_model_save_path_pickle + f'_in_subject_#{exm_subj}.pik', 'wb') as f:
                        pickle.dump(pickle.dumps(model), f)
                except:
                    pass
                model = model.to(device)
            waiting = 0
        else:
            waiting += 1

        # model saving
        if valid_acc_majority > tot_best_valid_acc:
            print(f'Total best val ACC {tot_best_valid_acc:.4f} -> {valid_acc_majority:.4f}')
            tot_best_valid_acc = valid_acc_majority
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path + f'_in_subject_#{exm_subj}_best_total.pt')
                try:
                    with open(Best_model_save_path_pickle + f'_in_subject_#{exm_subj}_best_total.pik', 'wb') as f:
                        pickle.dump(pickle.dumps(model), f)
                except:
                    pass
                model = model.to(device)
            waiting = 0
        else:
            waiting += 1

        # model early stopping
        if waiting >= patience:
            print(f'Stop training at epoch {epoch}')
            print(f'Best val loss : {best_valid_loss:.4f}')
            break
    try:
        # plt.figure()
        # plt.plot(range(len(lr_cur)), lr_cur)
        # plt.show(block=False)
        return best_model, history, analog_res_train, analog_res_val, best_valid_loss, best_val_acc_regular, best_val_acc_majority, best_train_acc, val_acc_regular, val_acc_majority, train_acc
    except:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        return best_model, history



def train_with_val_regular_and_majority_decision_without_transform_after_encoder(model, loaders, optimizer, criterion, n_epochs,
          patience, device, metric=None, Best_model_save_path=None, Best_model_save_path_pickle=None, args=None, best_valid_loss=np.inf, tr_subj=None, exm_subj=None, total_num_of_subjects=27, best_valid_acc=0, tot_best_valid_acc=0):
    """Training function.

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    metric : None | callable
        Metric to use to evaluate performance on the training and
        validation sets. Defaults to balanced accuracy.

    Returns
    -------
    best_model : instance of nn.Module
        The model that led to the best prediction on the validation
        dataset.
    history : list of dicts
        Training history (loss, accuracy, etc.)
    """
    # best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    waiting = 0
    history = list()
    lr_cur = []

    if metric is None:
        metric = balanced_accuracy_score
    model_enc = Encoder(num_features_input=196)
    print('epoch \t train_loss \t valid_loss \t train_perf \t valid_perf_regular \t valid_perf_majority \t train_acc \t valid_acc_regular \t valid_acc_majority \t train_confidence \t valid_confidence')
    print('--------------------------------------------------------------------------------------------------------------------------------------')
    loader_train = loaders['train_loader']
    loader_valid = loaders['val_loader']
    if args.lr_decay:
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        # scheduler = ReduceLROnPlateau(optimizer, 'min')
        decayRate = 0.975
        lambda2 = lambda epoch: decayRate ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    best_val_acc_regular = 0
    best_val_acc_majority = 0
    best_train_acc = 0

    val_acc_regular = np.zeros((n_epochs, 1))
    val_acc_majority = np.zeros((n_epochs, 1))
    train_acc = np.zeros((n_epochs, 1))

    for epoch in range(1, n_epochs + 1):
        train_loss, train_perf, train_acc_result, train_confidence, analog_res_train = _do_train_without_transform_after_encoder(
            model, model_enc, loader_train, optimizer, criterion, device, metric=metric, args=args)
        valid_loss, valid_perf_regular, valid_perf_majority, valid_acc_regular, valid_acc_majority, valid_confidence, analog_res_val = _validate_with_majority_and_regular_without_transform_after_encoder(
            model, model_enc, loader_valid, criterion, device, metric=metric, args=args)
        if args.lr_decay:
            scheduler.step()
            print(f'LR is: {scheduler.get_last_lr()} @ Epoch: {epoch}')
            lr_cur.append(scheduler.get_last_lr())

        val_acc_regular[epoch-1] = valid_acc_regular
        val_acc_majority[epoch - 1] = valid_acc_majority
        train_acc[epoch - 1] = train_acc_result

        if best_val_acc_regular <= valid_acc_regular:
            best_val_acc_regular = valid_acc_regular
            print('New val acc regular')
        if best_val_acc_majority <= valid_acc_majority:
            best_val_acc_majority = valid_acc_majority
            print('New val acc majority')
        if best_train_acc <= train_acc_result:
            best_train_acc = train_acc_result
        history.append(
            {'epoch': epoch,
             'train_loss': train_loss, 'valid_loss': valid_loss,
             'train_perf': train_perf, 'valid_perf_regular': valid_perf_regular, 'valid_perf_majority': valid_perf_majority,
             'train_acc': train_acc_result.item(), 'valid_acc_regular': valid_acc_regular.item(), 'valid_acc_majority': valid_acc_majority.item(),
             'train_confidence': train_confidence.item(), 'valid_confidence': valid_confidence.item(),
             'Exm subj': exm_subj, 'Trained subj': tr_subj})

        print(f'{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} '
              f'\t {train_perf:0.4f} \t {valid_perf_regular:0.4f}  \t  {valid_perf_majority:0.4f}'
              f'\t {train_acc_result:0.4f} \t {valid_acc_regular:0.4f}  \t  {valid_acc_majority:0.4f}'
              f'\t {train_confidence:0.4f} \t {valid_confidence:0.4f}  ')

        # model saving
        if valid_acc_majority > best_valid_acc:
            print(f'best val ACC {best_valid_acc:.4f} -> {valid_acc_majority:.4f}')
            best_valid_acc = valid_acc_majority
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path + f'_in_subject_#{exm_subj}.pt')
                try:
                    with open(Best_model_save_path_pickle + f'_in_subject_#{exm_subj}.pik', 'wb') as f:
                        pickle.dump(pickle.dumps(model), f)
                except:
                    pass
                model = model.to(device)
            waiting = 0
        else:
            waiting += 1

        # model saving
        if valid_acc_majority > tot_best_valid_acc:
            print(f'Total best val ACC {tot_best_valid_acc:.4f} -> {valid_acc_majority:.4f}')
            tot_best_valid_acc = valid_acc_majority
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path + f'_in_subject_#{exm_subj}_best_total.pt')
                try:
                    with open(Best_model_save_path_pickle + f'_in_subject_#{exm_subj}_best_total.pik', 'wb') as f:
                        pickle.dump(pickle.dumps(model), f)
                except:
                    pass
                model = model.to(device)
            waiting = 0
        else:
            waiting += 1

        # model early stopping
        if waiting >= patience:
            print(f'Stop training at epoch {epoch}')
            print(f'Best val loss : {best_valid_loss:.4f}')
            break
    try:
        # plt.figure()
        # plt.plot(range(len(lr_cur)), lr_cur)
        # plt.show(block=False)
        return best_model, history, analog_res_train, analog_res_val, best_valid_loss, best_val_acc_regular, best_val_acc_majority, best_train_acc, val_acc_regular, val_acc_majority, train_acc
    except:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        return best_model, history


def train_with_val_majority_decision(model, loaders, optimizer, criterion, n_epochs,
                                     patience, device, metric=None, Best_model_save_path=None,
                                     Best_model_save_path_pickle=None, args=None, best_valid_loss=np.inf,
                                     tr_subj=None, exm_subj=None):
    """Training function.

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    metric : None | callable
        Metric to use to evaluate performance on the training and
        validation sets. Defaults to balanced accuracy.

    Returns
    -------
    best_model : instance of nn.Module
        The model that led to the best prediction on the validation
        dataset.
    history : list of dicts
        Training history (loss, accuracy, etc.)
    """
    # best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    waiting = 0
    history = list()
    lr_cur = []

    if metric is None:
        metric = balanced_accuracy_score

    print(
        'epoch \t train_loss \t valid_loss \t train_perf \t valid_perf \t train_acc \t valid_acc \t train_confidence \t valid_confidence')
    print(
        '--------------------------------------------------------------------------------------------------------------------------------------')
    loader_train = loaders['train_loader']
    loader_valid = loaders['val_loader']
    if args.lr_decay:
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        # scheduler = ReduceLROnPlateau(optimizer, 'min')
        decayRate = 0.99
        lambda2 = lambda epoch: decayRate ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    best_val_acc = 0
    best_train_acc = 0
    for epoch in range(1, n_epochs + 1):
        train_loss, train_perf, train_acc, train_confidence, analog_res_train = _do_train(
            model, loader_train, optimizer, criterion, device, metric=metric, args=args)
        valid_loss, valid_perf, valid_acc, valid_confidence, analog_res_val = _validate_with_majority(
            model, loader_valid, criterion, device, metric=metric, args=args)
        if args.lr_decay:
            scheduler.step()
            print(f'LR is: {scheduler.get_lr()[0]} @ Epoch: {epoch}')
            lr_cur.append(scheduler.get_lr()[0])
        if best_val_acc <= valid_acc:
            best_val_acc = valid_acc
            print('New val acc')
        if best_train_acc <= train_acc:
            best_train_acc = train_acc
        history.append(
            {'epoch': epoch,
             'train_loss': train_loss, 'valid_loss': valid_loss,
             'train_perf': train_perf, 'valid_perf': valid_perf,
             'train_acc': train_acc.item(), 'valid_acc': valid_acc.item(),
             'train_confidence': train_confidence.item(), 'valid_confidence': valid_confidence.item(),
             'Exm subj': exm_subj, 'Trained subj': tr_subj})

        print(f'{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} '
              f'\t {train_perf:0.4f} \t {valid_perf:0.4f}  '
              f'\t {train_acc:0.4f} \t {valid_acc:0.4f}  '
              f'\t {train_confidence:0.4f} \t {valid_confidence:0.4f}  ')

        # model saving
        if valid_loss < best_valid_loss:
            print(f'best val loss {best_valid_loss:.4f} -> {valid_loss:.4f}')
            best_valid_loss = valid_loss
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path)
                try:
                    with open(Best_model_save_path_pickle, 'wb') as f:
                        pickle.dump(pickle.dumps(model), f)
                except:
                    pass
                model = model.to(device)
            waiting = 0
        else:
            waiting += 1

        # model early stopping
        if waiting >= patience:
            print(f'Stop training at epoch {epoch}')
            print(f'Best val loss : {best_valid_loss:.4f}')
            break
    try:
        # plt.figure()
        # plt.plot(range(len(lr_cur)), lr_cur)
        # plt.show(block=False)
        return best_model, history, analog_res_train, analog_res_val, best_valid_loss, best_val_acc, best_train_acc
    except:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        return best_model, history

def train_lstm(model, loaders, optimizer, criterion, n_epochs,
          patience, device, metric=None, Best_model_save_path=None, Best_model_save_path_pickle=None, args=None, best_valid_loss=np.inf):
    """Training function.

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    metric : None | callable
        Metric to use to evaluate performance on the training and
        validation sets. Defaults to balanced accuracy.

    Returns
    -------
    best_model : instance of nn.Module
        The model that led to the best prediction on the validation
        dataset.
    history : list of dicts
        Training history (loss, accuracy, etc.)
    """
    # best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    waiting = 0
    history = list()
    lr_cur = []

    if metric is None:
        metric = balanced_accuracy_score

    print('epoch \t train_loss \t valid_loss \t train_perf \t valid_perf \t train_acc \t valid_acc \t train_confidence \t valid_confidence')
    print('--------------------------------------------------------------------------------------------------------------------------------------')
    loader_train = loaders['train_loader']
    loader_valid = loaders['val_loader']
    if args.lr_decay:
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        # scheduler = ReduceLROnPlateau(optimizer, 'min')
        decayRate = 0.98
        lambda2 = lambda epoch: decayRate ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
        hn = cn = None
    for epoch in range(1, n_epochs + 1):
        train_loss, train_perf, train_acc, train_confidence, analog_res_train, hn, cn = _do_train_lstm(
            model, loader_train, optimizer, criterion, device, metric=metric, args=args, hn=hn, cn=cn)
        valid_loss, valid_perf, valid_acc, valid_confidence, analog_res_val = _validate_lstm(
            model, loader_valid, criterion, device, metric=metric, args=args, hn=hn, cn=cn)
        if args.lr_decay:
            scheduler.step()
            print(f'LR is: {scheduler.get_lr()[0]} @ Epoch: {epoch}')
            lr_cur.append(scheduler.get_lr()[0])
        history.append(
            {'epoch': epoch,
             'train_loss': train_loss, 'valid_loss': valid_loss,
             'train_perf': train_perf, 'valid_perf': valid_perf,
             'train_acc': train_acc.item(), 'valid_acc': valid_acc.item(),
             'train_confidence': train_confidence.item(), 'valid_confidence': valid_confidence.item()})

        print(f'{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} '
              f'\t {train_perf:0.4f} \t {valid_perf:0.4f}  '
              f'\t {train_acc:0.4f} \t {valid_acc:0.4f}  '
              f'\t {train_confidence:0.4f} \t {valid_confidence:0.4f}  ')

        # model saving
        if valid_loss < best_valid_loss:
            print(f'best val loss {best_valid_loss:.4f} -> {valid_loss:.4f}')
            best_valid_loss = valid_loss
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path)
                try:
                    with open(Best_model_save_path_pickle, 'wb') as f:
                        pickle.dump(pickle.dumps(model), f)
                except:
                    pass
                model = model.to(device)
            waiting = 0
        else:
            waiting += 1

        # model early stopping
        if waiting >= patience:
            print(f'Stop training at epoch {epoch}')
            print(f'Best val loss : {best_valid_loss:.4f}')
            break
    try:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        return best_model, history, analog_res_train, analog_res_val, best_valid_loss
    except:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        return best_model, history

def train_AttenSleep(model, loaders, optimizer, criterion, n_epochs,
          patience, device, metric=None, Best_model_save_path=None, Best_model_save_path_pickle=None, args=None, best_valid_loss=np.inf):
    """Training function.

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    metric : None | callable
        Metric to use to evaluate performance on the training and
        validation sets. Defaults to balanced accuracy.

    Returns
    -------
    best_model : instance of nn.Module
        The model that led to the best prediction on the validation
        dataset.
    history : list of dicts
        Training history (loss, accuracy, etc.)
    """
    # best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    waiting = 0
    history = list()
    lr_cur = []

    if metric is None:
        metric = balanced_accuracy_score

    print('epoch \t train_loss \t valid_loss \t train_perf \t valid_perf \t train_acc \t valid_acc \t train_confidence \t valid_confidence')
    print('--------------------------------------------------------------------------------------------------------------------------------------')
    loader_train = loaders['train_loader']
    loader_valid = loaders['val_loader']
    if args.lr_decay:
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        # scheduler = ReduceLROnPlateau(optimizer, 'min')
        decayRate = 0.985
        lambda2 = lambda epoch: decayRate ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    for epoch in range(1, n_epochs + 1):
        train_loss, train_perf, train_acc, train_confidence, analog_res_train = _do_train_AttenSleep(
            model, loader_train, optimizer, criterion, device, metric=metric, args=args)
        valid_loss, valid_perf, valid_acc, valid_confidence, analog_res_val = _validate_AttenSleep(
            model, loader_valid, criterion, device, metric=metric, args=args)
        if args.lr_decay:
            scheduler.step()
            print(f'LR is: {scheduler.get_lr()[0]} @ Epoch: {epoch}')
            lr_cur.append(scheduler.get_lr()[0])
        history.append(
            {'epoch': epoch,
             'train_loss': train_loss, 'valid_loss': valid_loss,
             'train_perf': train_perf, 'valid_perf': valid_perf,
             'train_acc': train_acc.item(), 'valid_acc': valid_acc.item(),
             'train_confidence': train_confidence.item(), 'valid_confidence': valid_confidence.item()})

        print(f'{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} '
              f'\t {train_perf:0.4f} \t {valid_perf:0.4f}  '
              f'\t {train_acc:0.4f} \t {valid_acc:0.4f}  '
              f'\t {train_confidence:0.4f} \t {valid_confidence:0.4f}  ')

        # model saving
        if valid_loss < best_valid_loss:
            print(f'best val loss {best_valid_loss:.4f} -> {valid_loss:.4f}')
            best_valid_loss = valid_loss
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path)
                try:
                    with open(Best_model_save_path_pickle, 'wb') as f:
                        pickle.dump(pickle.dumps(model), f)
                except:
                    pass
                model = model.to(device)
            waiting = 0
        else:
            waiting += 1

        # model early stopping
        if waiting >= patience:
            print(f'Stop training at epoch {epoch}')
            print(f'Best val loss : {best_valid_loss:.4f}')
            break
    try:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        return best_model, history, analog_res_train, analog_res_val, best_valid_loss
    except:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        return best_model, history

def train_VAE(model, loaders, optimizer, criterion, n_epochs,
          patience, device, metric=None, Best_model_save_path=None, Best_model_save_path_pickle=None, args=None, best_valid_loss=np.inf):
    """Training function.

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    metric : None | callable
        Metric to use to evaluate performance on the training and
        validation sets. Defaults to balanced accuracy.

    Returns
    -------
    best_model : instance of nn.Module
        The model that led to the best prediction on the validation
        dataset.
    history : list of dicts
        Training history (loss, accuracy, etc.)
    """
    # best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    waiting = 0
    history = list()
    lr_cur = []

    if metric is None:
        metric = balanced_accuracy_score

    print('epoch \t train_loss \t valid_loss \t train_perf \t valid_perf \t train_acc \t valid_acc \t train_confidence \t valid_confidence')
    print('--------------------------------------------------------------------------------------------------------------------------------------')
    loader_train = loaders['train_loader']
    loader_valid = loaders['val_loader']
    if args.lr_decay:
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        # scheduler = ReduceLROnPlateau(optimizer, 'min')
        decayRate = 0.93
        lambda2 = lambda epoch: decayRate ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    for epoch in range(1, n_epochs + 1):
        train_loss, train_perf, train_acc, train_confidence, analog_res_train = _do_train_VAE(
            model, loader_train, optimizer, criterion, device, metric=metric, args=args)
        valid_loss, valid_perf, valid_acc, valid_confidence, analog_res_val = _validate_VAE(
            model, loader_valid, criterion, device, metric=metric, args=args)
        if args.lr_decay:
            scheduler.step()
            print(f'LR is: {scheduler.get_lr()[0]} @ Epoch: {epoch}')
            lr_cur.append(scheduler.get_lr()[0])
        history.append(
            {'epoch': epoch,
             'train_loss': train_loss, 'valid_loss': valid_loss,
             'train_perf': train_perf, 'valid_perf': valid_perf,
             'train_acc': train_acc.item(), 'valid_acc': valid_acc.item(),
             'train_confidence': train_confidence.item(), 'valid_confidence': valid_confidence.item()})

        print(f'{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} '
              f'\t {train_perf:0.4f} \t {valid_perf:0.4f}  '
              f'\t {train_acc:0.4f} \t {valid_acc:0.4f}  '
              f'\t {train_confidence:0.4f} \t {valid_confidence:0.4f}  ')

        # model saving
        if valid_loss < best_valid_loss:
            print(f'best val loss {best_valid_loss:.4f} -> {valid_loss:.4f}')
            best_valid_loss = valid_loss
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path)
                try:
                    with open(Best_model_save_path_pickle, 'wb') as f:
                        pickle.dump(pickle.dumps(model), f)
                except:
                    pass
                model = model.to(device)
            waiting = 0
        else:
            waiting += 1

        # model early stopping
        if waiting >= patience:
            print(f'Stop training at epoch {epoch}')
            print(f'Best val loss : {best_valid_loss:.4f}')
            break
    try:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        return best_model, history, analog_res_train, analog_res_val, best_valid_loss
    except:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        return best_model, history



def train_subgroup_tSNE(model, loaders, optimizer, criterion, n_epochs,
          patience, device, metric=None, Best_model_save_path=None, Best_model_save_path_pickle=None, args=None, best_valid_loss=np.inf):
    """Training function.

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    metric : None | callable
        Metric to use to evaluate performance on the training and
        validation sets. Defaults to balanced accuracy.

    Returns
    -------
    best_model : instance of nn.Module
        The model that led to the best prediction on the validation
        dataset.
    history : list of dicts
        Training history (loss, accuracy, etc.)
    """
    # best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    waiting = 0
    history = list()
    lr_cur = []

    if metric is None:
        metric = balanced_accuracy_score

    print('epoch \t train_loss \t valid_loss \t train_perf \t valid_perf \t train_acc \t valid_acc \t train_confidence \t valid_confidence')
    print('--------------------------------------------------------------------------------------------------------------------------------------')
    loader_train = loaders['train_loader']
    loader_valid = loaders['val_loader']
    loader_train_tsne = loaders['t_SNE_loader']
    if args.lr_decay:
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        # scheduler = ReduceLROnPlateau(optimizer, 'min')
        decayRate = 0.93
        lambda2 = lambda epoch: decayRate ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    for epoch in range(1, n_epochs + 1):
        train_loss, train_perf, train_acc, train_confidence, analog_res_train = _do_train_subgroup_t_SNE(
            model, loader_train, loader_train_tsne, optimizer, criterion, device, metric=metric, args=args)
        valid_loss, valid_perf, valid_acc, valid_confidence, analog_res_val = _validate_subgroup_t_SNE(
            model, loader_valid, loader_train_tsne, criterion, device, metric=metric, args=args)
        if args.lr_decay:
            scheduler.step()
            print(f'LR is: {scheduler.get_lr()[0]} @ Epoch: {epoch}')
            lr_cur.append(scheduler.get_lr()[0])
        history.append(
            {'epoch': epoch,
             'train_loss': train_loss, 'valid_loss': valid_loss,
             'train_perf': train_perf, 'valid_perf': valid_perf,
             'train_acc': train_acc.item(), 'valid_acc': valid_acc.item(),
             'train_confidence': train_confidence.item(), 'valid_confidence': valid_confidence.item()})

        print(f'{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} '
              f'\t {train_perf:0.4f} \t {valid_perf:0.4f}  '
              f'\t {train_acc:0.4f} \t {valid_acc:0.4f}  '
              f'\t {train_confidence:0.4f} \t {valid_confidence:0.4f}  ')

        # model saving
        if valid_loss < best_valid_loss:
            print(f'best val loss {best_valid_loss:.4f} -> {valid_loss:.4f}')
            best_valid_loss = valid_loss
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path)
                try:
                    with open(Best_model_save_path_pickle, 'wb') as f:
                        pickle.dump(pickle.dumps(model), f)
                except:
                    pass
                model = model.to(device)
            waiting = 0
        else:
            waiting += 1

        # model early stopping
        if waiting >= patience:
            print(f'Stop training at epoch {epoch}')
            print(f'Best val loss : {best_valid_loss:.4f}')
            break
    try:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        return best_model, history, analog_res_train, analog_res_val, best_valid_loss
    except:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        return best_model, history

def train_subgroup_tSNE_with_torch(model, loaders, optimizer, criterion, n_epochs,
          patience, device, metric=None, Best_model_save_path=None, Best_model_save_path_pickle=None, args=None, best_valid_loss=np.inf):
    """Training function.

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    metric : None | callable
        Metric to use to evaluate performance on the training and
        validation sets. Defaults to balanced accuracy.

    Returns
    -------
    best_model : instance of nn.Module
        The model that led to the best prediction on the validation
        dataset.
    history : list of dicts
        Training history (loss, accuracy, etc.)
    """
    # best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    waiting = 0
    history = list()
    lr_cur = []

    if metric is None:
        metric = balanced_accuracy_score

    print('epoch \t train_loss \t valid_loss \t train_perf \t valid_perf \t train_acc \t valid_acc \t train_confidence \t valid_confidence')
    print('--------------------------------------------------------------------------------------------------------------------------------------')
    loader_train = loaders['train_loader']
    loader_valid = loaders['val_loader']
    loader_train_tsne = loaders['t_SNE_loader']
    if args.lr_decay:
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        # scheduler = ReduceLROnPlateau(optimizer, 'min')
        decayRate = 0.98
        lambda2 = lambda epoch: decayRate ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    for epoch in range(1, n_epochs + 1):
        train_loss, train_perf, train_acc, train_confidence, analog_res_train = _do_train_subgroup_t_SNE_with_torch(
            model, loader_train, loader_train_tsne, optimizer, criterion, device, metric=metric, args=args)
        valid_loss, valid_perf, valid_acc, valid_confidence, analog_res_val = _validate_subgroup_t_SNE_with_torch(
            model, loader_valid, loader_train_tsne, criterion, device, metric=metric, args=args)
        if args.lr_decay:
            scheduler.step()
            print(f'LR is: {scheduler.get_lr()[0]} @ Epoch: {epoch}')
            lr_cur.append(scheduler.get_lr()[0])
        history.append(
            {'epoch': epoch,
             'train_loss': train_loss, 'valid_loss': valid_loss,
             'train_perf': train_perf, 'valid_perf': valid_perf,
             'train_acc': train_acc.item(), 'valid_acc': valid_acc.item(),
             'train_confidence': train_confidence.item(), 'valid_confidence': valid_confidence.item()})

        print(f'{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} '
              f'\t {train_perf:0.4f} \t {valid_perf:0.4f}  '
              f'\t {train_acc:0.4f} \t {valid_acc:0.4f}  '
              f'\t {train_confidence:0.4f} \t {valid_confidence:0.4f}  ')

        # model saving
        if valid_loss < best_valid_loss:
            print(f'best val loss {best_valid_loss:.4f} -> {valid_loss:.4f}')
            best_valid_loss = valid_loss
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path)
                try:
                    with open(Best_model_save_path_pickle, 'wb') as f:
                        pickle.dump(pickle.dumps(model), f)
                except:
                    pass
                model = model.to(device)
            waiting = 0
        else:
            waiting += 1

        # model early stopping
        if waiting >= patience:
            print(f'Stop training at epoch {epoch}')
            print(f'Best val loss : {best_valid_loss:.4f}')
            break
    try:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        return best_model, history, analog_res_train, analog_res_val, best_valid_loss
    except:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        return best_model, history

def train_optimization(model, loaders, optimizer, criterion, n_epochs,
          patience, device, metric=None, Best_model_save_path=None, Best_model_save_path_pickle=None, args=None, best_valid_loss=np.inf, decayRate=0.955):
    """Training function.

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    metric : None | callable
        Metric to use to evaluate performance on the training and
        validation sets. Defaults to balanced accuracy.

    Returns
    -------
    best_model : instance of nn.Module
        The model that led to the best prediction on the validation
        dataset.
    history : list of dicts
        Training history (loss, accuracy, etc.)
    """
    # best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    waiting = 0
    history = list()
    lr_cur = []

    if metric is None:
        metric = balanced_accuracy_score

    print('epoch \t train_loss \t valid_loss \t train_perf \t valid_perf \t train_acc \t valid_acc \t train_confidence \t valid_confidence')
    print('--------------------------------------------------------------------------------------------------------------------------------------')
    loader_train = loaders['train_loader']
    loader_valid = loaders['val_loader']
    if args.lr_decay:
        # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        # scheduler = ReduceLROnPlateau(optimizer, 'min')
        decayRate = 0.93
        lambda2 = lambda epoch: decayRate ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    avg_valid_acc = 0
    for epoch in range(1, n_epochs + 1):
        train_loss, train_perf, train_acc, train_confidence, analog_res_train = _do_train(
            model, loader_train, optimizer, criterion, device, metric=metric, args=args)
        valid_loss, valid_perf, valid_acc, valid_confidence, analog_res_val = _validate(
            model, loader_valid, criterion, device, metric=metric, args=args)
        if args.lr_decay:
            scheduler.step()
            print(f'LR is: {scheduler.get_lr()[0]} @ Epoch: {epoch}')
            lr_cur.append(scheduler.get_lr()[0])
        history.append(
            {'epoch': epoch,
             'train_loss': train_loss, 'valid_loss': valid_loss,
             'train_perf': train_perf, 'valid_perf': valid_perf,
             'train_acc': train_acc.item(), 'valid_acc': valid_acc.item(),
             'train_confidence': train_confidence.item(), 'valid_confidence': valid_confidence.item()})

        print(f'{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} '
              f'\t {train_perf:0.4f} \t {valid_perf:0.4f}  '
              f'\t {train_acc:0.4f} \t {valid_acc:0.4f}  '
              f'\t {train_confidence:0.4f} \t {valid_confidence:0.4f}  ')

        # model saving
        if valid_loss < best_valid_loss:
            print(f'best val loss {best_valid_loss:.4f} -> {valid_loss:.4f}')
            best_valid_loss = valid_loss
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path)
                try:
                    with open(Best_model_save_path_pickle, 'wb') as f:
                        pickle.dump(pickle.dumps(model), f)
                except:
                    pass
                model = model.to(device)
            waiting = 0
        else:
            waiting += 1

        # model early stopping
        if waiting >= patience:
            print(f'Stop training at epoch {epoch}')
            print(f'Best val loss : {best_valid_loss:.4f}')
            break
        avg_valid_acc += valid_acc.item()
    avg_valid_acc /= n_epochs

    try:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        return best_model, history, analog_res_train, analog_res_val, best_valid_loss, avg_valid_acc
    except:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        return best_model, history


# UD technique

def train_with_esn(model, loaders, optimizer, criterion, n_epochs,
          patience, device, metric=None, Best_model_save_path=None, Best_model_save_path_pickle=None, args=None, best_valid_loss=np.inf):
    """Training function.

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    metric : None | callable
        Metric to use to evaluate performance on the training and
        validation sets. Defaults to balanced accuracy.

    Returns
    -------
    best_model : instance of nn.Module
        The model that led to the best prediction on the validation
        dataset.
    history : list of dicts
        Training history (loss, accuracy, etc.)
    """
    # best_valid_loss = np.inf
    # best_model = copy.deepcopy(model)
    waiting = 0
    history = list()
    lr_cur = []

    if metric is None:
        metric = balanced_accuracy_score

    print('epoch \t train_loss \t valid_loss \t train_perf \t valid_perf \t train_acc \t valid_acc \t train_confidence \t valid_confidence')
    print('--------------------------------------------------------------------------------------------------------------------------------------')
    loader_train = loaders['train_loader']
    loader_valid = loaders['val_loader']
    if args.lr_decay:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_rate, weight_decay=1e-4)
        # scheduler = StepLR(optimizer, step_size=10, gamma=1e-4)
        decayRate = 0.97
        lambda2 = lambda epoch: decayRate ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
        # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
        # scheduler = ReduceLROnPlateau(optimizer, 'min')
        # decayRate = 0.98
        # lambda2 = lambda epoch: decayRate ** epoch
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    for epoch in range(1, n_epochs + 1):
        train_loss, train_perf, train_acc, train_confidence, analog_res_train, hidden, cur_best_model = _do_train_with_esn(
            model, loader_train, optimizer, criterion, device, metric=metric, args=args)
        valid_loss, valid_perf, valid_acc, valid_confidence, analog_res_val = _validate_with_esn(
            cur_best_model, loader_valid, criterion, device, metric=metric, args=args, hidden=hidden)
        if args.lr_decay:
            try:
                scheduler.step()
                print(f'LR is: {scheduler.get_lr()[0]} @ Epoch: {epoch}')
                lr_cur.append(scheduler.get_lr()[0])
            except:
                pass
            # print(f'LR is: {scheduler.get_lr()[0]} @ Epoch: {epoch}')
            # lr_cur.append(scheduler.get_lr()[0])
        history.append(
            {'epoch': epoch,
             'train_loss': train_loss, 'valid_loss': valid_loss,
             'train_perf': train_perf, 'valid_perf': valid_perf,
             'train_acc': train_acc.item(), 'valid_acc': valid_acc.item(),
             'train_confidence': train_confidence.item(), 'valid_confidence': valid_confidence.item()})

        print(f'{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} '
              f'\t {train_perf:0.4f} \t {valid_perf:0.4f}  '
              f'\t {train_acc:0.4f} \t {valid_acc:0.4f}  '
              f'\t {train_confidence:0.4f} \t {valid_confidence:0.4f}  ')

        # model saving
        if valid_loss < best_valid_loss:
            print(f'best val loss {best_valid_loss:.4f} -> {valid_loss:.4f}')
            best_valid_loss = valid_loss
            best_hidden = hidden
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                best_model = copy.deepcopy(model)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path)
                try:
                    with open(Best_model_save_path_pickle, 'wb') as f:
                        pickle.dump(pickle.dumps(model), f)
                except:
                    pass
                model = model.to(device)
            waiting = 0
        else:
            waiting += 1

        # model early stopping
        if waiting >= patience:
            print(f'Stop training at epoch {epoch}')
            print(f'Best val loss : {best_valid_loss:.4f}')
            break
    try:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        try:
            return best_model, history, analog_res_train, analog_res_val, best_valid_loss, best_hidden
        except:
            return best_model, history, analog_res_train, analog_res_val, best_valid_loss
    except:
        plt.figure()
        plt.plot(range(len(lr_cur)), lr_cur)
        plt.show(block=False)
        try:
            return best_model, history, best_hidden
        except:
            return best_model, history

def _do_train_UD(model, loader, optimizer, criterion, device, metric, args=None):
    confidence = 0
    # training loop
    model.train()
    train_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes + 1)))
    else:
        analog_res = []

    for idx_batch, (batch_x, batch_y) in enumerate(loader):
        optimizer.zero_grad()
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.int64)

        output = model(batch_x)
        confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
        confidence += torch.mean(confi)
        loss = criterion(output, batch_y)
        # loss = my_loss(output, batch_y)
        # cur_analog_res = [output, batch_y]
        analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), 0:args.n_classes] = (
            F.softmax(output, dim=1)).cpu().detach().numpy()
        analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)),
        args.n_classes] = batch_y.cpu().detach().numpy()

        Output = F.softmax(output).detach().cpu().numpy()
        indices = (-Output).argsort()
        minus_vals_logic = (indices[:, 0] > indices[:, 1])
        plus_vals_logic = (indices[:, 0] < indices[:, 1])
        OUTPUT = Output[:, 0]
        OUTPUT[minus_vals_logic] = indices[minus_vals_logic, 0] - (
                Output[minus_vals_logic, indices[minus_vals_logic, 1]] / Output[
            minus_vals_logic, indices[minus_vals_logic, 0]])
        OUTPUT[plus_vals_logic] = indices[plus_vals_logic, 0] + (
                Output[plus_vals_logic, indices[plus_vals_logic, 1]] / Output[
            plus_vals_logic, indices[plus_vals_logic, 0]])
        OUTPUT = np.abs(OUTPUT)


        loss.backward()
        optimizer.step()

        y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
        y_true_all.append(batch_y.cpu().numpy())

        train_loss[idx_batch] = loss.item()

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader)
    try:
        return np.mean(train_loss), perf, acc, confidence, analog_res
    except:
        return np.mean(train_loss), perf, acc, confidence


def _validate_UD(model, loader, criterion, device, metric, args=None):
    confidence = 0
    # validation loop
    model.eval()

    val_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes + 1)))
    else:
        analog_res = []
    with torch.no_grad():
        for idx_batch, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.int64)
            output = model.forward(batch_x)

            confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
            confidence += torch.mean(confi)
            loss = criterion(output, batch_y)
            # loss = my_loss(output, batch_y)
            # cur_analog_res = [output, batch_y]
            # analog_res.append(cur_analog_res)
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), 0:args.n_classes] = (
                F.softmax(output, dim=1)).cpu().detach().numpy()
            analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)),
            args.n_classes] = batch_y.cpu().detach().numpy()
            val_loss[idx_batch] = loss.item()

            y_pred_all.append(torch.argmax(output, axis=1).cpu().numpy())
            y_true_all.append(batch_y.cpu().numpy())

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader)

    try:
        return np.mean(val_loss), perf, acc, confidence, analog_res
    except:
        return np.mean(val_loss), perf, acc, confidence


def train_kfold_UD(model, loaders, optimizer, criterion_train, criterion_val, n_epochs,
                patience, device, metric=None, Best_model_save_path=None, args=None):
    """Training function.

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    metric : None | callable
        Metric to use to evaluate performance on the training and
        validation sets. Defaults to balanced accuracy.

    Returns
    -------
    best_model : instance of nn.Module
        The model that led to the best prediction on the validation
        dataset.
    history : list of dicts
        Training history (loss, accuracy, etc.)
    """
    best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    waiting = 0
    history = list()

    if metric is None:
        metric = balanced_accuracy_score

    print(
        'epoch \t train_loss \t valid_loss \t train_perf \t valid_perf \t train_acc \t valid_acc \t train_confidence \t valid_confidence')
    print(
        '--------------------------------------------------------------------------------------------------------------------------------------')
    loader_train = loaders['train_loader']
    loader_valid = loaders['val_loader']
    for epoch in range(1, n_epochs + 1):
        train_loss, train_perf, train_acc, train_confidence, analog_res_train = _do_train(
            model, loader_train, optimizer, criterion_train, device, metric=metric, args=args)
        valid_loss, valid_perf, valid_acc, valid_confidence, analog_res_val = _validate(
            model, loader_valid, criterion_val, device, metric=metric, args=args)
        history.append(
            {'epoch': epoch,
             'train_loss': train_loss, 'valid_loss': valid_loss,
             'train_perf': train_perf, 'valid_perf': valid_perf,
             'train_acc': train_acc.item(), 'valid_acc': valid_acc.item(),
             'train_confidence': train_confidence.item(), 'valid_confidence': valid_confidence.item()})

        print(f'{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} '
              f'\t {train_perf:0.4f} \t {valid_perf:0.4f}  '
              f'\t {train_acc:0.4f} \t {valid_acc:0.4f}  '
              f'\t {train_confidence:0.4f} \t {valid_confidence:0.4f}  ')

        # model saving
        if valid_loss < best_valid_loss:
            print(f'best val loss {best_valid_loss:.4f} -> {valid_loss:.4f}')
            best_valid_loss = valid_loss
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path)
            waiting = 0
        else:
            waiting += 1

        # model early stopping
        if waiting >= patience:
            print(f'Stop training at epoch {epoch}')
            print(f'Best val loss : {best_valid_loss:.4f}')
            break
    try:
        return best_model, history, analog_res_train, analog_res_val
    except:
        return best_model, history



def train_UD(model, loaders, optimizer, criterion, n_epochs,
          patience, device, metric=None, Best_model_save_path=None, Best_model_save_path_pickle=None, args=None,
          best_valid_loss=np.inf):
    """Training function.

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    metric : None | callable
        Metric to use to evaluate performance on the training and
        validation sets. Defaults to balanced accuracy.

    Returns
    -------
    best_model : instance of nn.Module
        The model that led to the best prediction on the validation
        dataset.
    history : list of dicts
        Training history (loss, accuracy, etc.)
    """
    # best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    waiting = 0
    history = list()

    if metric is None:
        metric = balanced_accuracy_score

    print(
        'epoch \t train_loss \t valid_loss \t train_perf \t valid_perf \t train_acc \t valid_acc \t train_confidence \t valid_confidence')
    print(
        '--------------------------------------------------------------------------------------------------------------------------------------')
    loader_train = loaders['train_loader']
    loader_valid = loaders['val_loader']
    for epoch in range(1, n_epochs + 1):
        train_loss, train_perf, train_acc, train_confidence, analog_res_train = _do_train_UD(
            model, loader_train, optimizer, criterion, device, metric=metric, args=args)
        valid_loss, valid_perf, valid_acc, valid_confidence, analog_res_val = _validate_UD(
            model, loader_valid, criterion, device, metric=metric, args=args)
        history.append(
            {'epoch': epoch,
             'train_loss': train_loss, 'valid_loss': valid_loss,
             'train_perf': train_perf, 'valid_perf': valid_perf,
             'train_acc': train_acc.item(), 'valid_acc': valid_acc.item(),
             'train_confidence': train_confidence.item(), 'valid_confidence': valid_confidence.item()})

        print(f'{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} '
              f'\t {train_perf:0.4f} \t {valid_perf:0.4f}  '
              f'\t {train_acc:0.4f} \t {valid_acc:0.4f}  '
              f'\t {train_confidence:0.4f} \t {valid_confidence:0.4f}  ')

        # model saving
        if valid_loss < best_valid_loss:
            print(f'best val loss {best_valid_loss:.4f} -> {valid_loss:.4f}')
            best_valid_loss = valid_loss
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path)
                try:
                    with open(Best_model_save_path_pickle, 'wb') as f:
                        pickle.dump(pickle.dumps(model), f)
                except:
                    pass
                model = model.to(device)
            waiting = 0
        else:
            waiting += 1

        # model early stopping
        if waiting >= patience:
            print(f'Stop training at epoch {epoch}')
            print(f'Best val loss : {best_valid_loss:.4f}')
            break
    try:
        return best_model, history, analog_res_train, analog_res_val, best_valid_loss
    except:
        return best_model, history



def train_with_macro_batch(model, loaders, optimizer, criterion, n_epochs,
          patience, device, metric=None, Best_model_save_path=None, Best_model_save_path_pickle=None, args=None, best_valid_loss=np.inf):
    """Training function.

    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader_train : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    metric : None | callable
        Metric to use to evaluate performance on the training and
        validation sets. Defaults to balanced accuracy.

    Returns
    -------
    best_model : instance of nn.Module
        The model that led to the best prediction on the validation
        dataset.
    history : list of dicts
        Training history (loss, accuracy, etc.)
    """
    # best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    waiting = 0
    history = list()

    if metric is None:
        metric = balanced_accuracy_score

    print('epoch \t train_loss \t valid_loss \t train_perf \t valid_perf \t train_acc \t valid_acc \t train_confidence \t valid_confidence')
    print('--------------------------------------------------------------------------------------------------------------------------------------')
    loader_train = loaders['train_loader']
    loader_valid = loaders['val_loader']
    if args.lr_decay:
        scheduler = StepLR(optimizer, step_size=35, gamma=0.2)
    for epoch in range(1, n_epochs + 1):
        train_loss, train_perf, train_acc, train_confidence, analog_res_train = _do_train_with_macro_batch(
            model, loader_train, optimizer, criterion, device, metric=metric, args=args)
        valid_loss, valid_perf, valid_acc, valid_confidence, analog_res_val = _validate_with_macro_batch(
            model, loader_valid, criterion, device, metric=metric, args=args)
        if args.lr_decay:
            scheduler.step()
            print(f'LR is: {scheduler.get_lr()[0]} @ Epoch: {epoch}')
        history.append(
            {'epoch': epoch,
             'train_loss': train_loss, 'valid_loss': valid_loss,
             'train_perf': train_perf, 'valid_perf': valid_perf,
             'train_acc': train_acc.item(), 'valid_acc': valid_acc.item(),
             'train_confidence': train_confidence.item(), 'valid_confidence': valid_confidence.item()})

        print(f'{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} '
              f'\t {train_perf:0.4f} \t {valid_perf:0.4f}  '
              f'\t {train_acc:0.4f} \t {valid_acc:0.4f}  '
              f'\t {train_confidence:0.4f} \t {valid_confidence:0.4f}  ')

        # model saving
        if valid_loss < best_valid_loss:
            print(f'best val loss {best_valid_loss:.4f} -> {valid_loss:.4f}')
            best_valid_loss = valid_loss
            # Save the best model
            if Best_model_save_path is None:
                best_model = copy.deepcopy(model)
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.to('cpu').state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'train_loss': train_loss,
                }, Best_model_save_path)
                try:
                    with open(Best_model_save_path_pickle, 'wb') as f:
                        pickle.dump(pickle.dumps(model), f)
                except:
                    pass
                model = model.to(device)
            waiting = 0
        else:
            waiting += 1

        # model early stopping
        if waiting >= patience:
            print(f'Stop training at epoch {epoch}')
            print(f'Best val loss : {best_valid_loss:.4f}')
            break
    try:
        return best_model, history, analog_res_train, analog_res_val, best_valid_loss
    except:
        return best_model, history


def _do_train_with_macro_batch(model, loader, optimizer, criterion, device, metric, args=None):
    confidence = 0
    # training loop
    model.train()
    train_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []
    flag = 0
    for idx_batch, (batch_x, batch_y) in enumerate(loader):
        if idx_batch == 0:
            num_of_smaples = batch_y.shape[1]
        optimizer.zero_grad()
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.int64)
        loss_tmp = 0
        model_outputs = np.zeros((batch_y.shape[0], batch_y.shape[1]), dtype=int)
        if batch_x.shape[1] < num_of_smaples:
            continue
        for ii in range(batch_y.shape[1]):
            try:
                output = model(torch.squeeze(batch_x[:, ii, :, :], dim=1))
            except:
                flag = 1
                continue
            confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
            confidence += torch.mean(confi) / batch_y.shape[1]
            loss_tmp += criterion(output, batch_y[:, ii])
            # loss = my_loss(output, batch_y)
            # cur_analog_res = [output, batch_y]
            analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), 0:args.n_classes] += ((F.softmax(output, dim=1)).cpu().detach().numpy() / batch_y.shape[1])
            analog_res[(idx_batch * len(batch_y)) : ((idx_batch+1) * len(batch_y)), args.n_classes] += (batch_y[:, ii].cpu().detach().numpy() / batch_y.shape[1])
            model_outputs[:, ii] = torch.argmax(output, axis=1).cpu().numpy().astype(int)

        if flag:
            flag = 0
            continue
        loss = loss_tmp
        loss.backward()
        optimizer.step()

        preds = np.zeros((batch_y.shape[0], 1))
        for kk in range(batch_y.shape[0]):
            counts = np.bincount(model_outputs[kk, :])
            preds[kk] = np.argmax(counts)

        y_pred_all.append(preds[:, 0])
        y_true_all.append(batch_y[:, -1].cpu().numpy())

        train_loss[idx_batch] = loss.item()

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader)
    try:
        return np.mean(train_loss), perf, acc, confidence, analog_res
    except:
        return np.mean(train_loss), perf, acc, confidence

def _validate_with_macro_batch(model, loader, criterion, device, metric, args=None):
    confidence = 0
    flag = 0
    # validation loop
    model.eval()

    val_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    if args is not None:
        analog_res = np.zeros((len(loader) * args.batch_size, (args.n_classes+1)))
    else:
        analog_res = []
    with torch.no_grad():
        for idx_batch, (batch_x, batch_y) in enumerate(loader):
            if idx_batch == 0:
                num_of_smaples = batch_y.shape[1]
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.int64)
            loss_tmp = 0
            model_outputs = np.zeros((batch_y.shape[0], batch_y.shape[1]), dtype=int)
            if batch_x.shape[1] < num_of_smaples:
                continue
            for ii in range(batch_y.shape[1]):
                try:
                    output = model(torch.squeeze(batch_x[:, ii, :, :], dim=1))
                except:
                    flag = 1
                    continue

                confi, _ = torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)
                confidence += torch.mean(confi) / batch_y.shape[1]
                loss_tmp += criterion(output, batch_y[:, ii])
                # loss = my_loss(output, batch_y)
                # cur_analog_res = [output, batch_y]
                # analog_res.append(cur_analog_res)
                analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), 0:args.n_classes] += (
                            (F.softmax(output, dim=1)).cpu().detach().numpy() / batch_y.shape[1])
                analog_res[(idx_batch * len(batch_y)): ((idx_batch + 1) * len(batch_y)), args.n_classes] += (
                            batch_y[:, ii].cpu().detach().numpy() / batch_y.shape[1])
                model_outputs[:, ii] = torch.argmax(output, axis=1).cpu().numpy().astype(int)

            if flag:
                flag = 0
                continue
            loss = loss_tmp
            val_loss[idx_batch] = loss.item()

            preds = np.zeros((batch_y.shape[0], 1))
            for kk in range(batch_y.shape[0]):
                counts = np.bincount(model_outputs[kk, :])
                preds[kk] = np.argmax(counts)

            y_pred_all.append(preds[:, 0])
            y_true_all.append(batch_y[:, -1].cpu().numpy())

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)
    perf = metric(y_true, y_pred)
    acc = np.mean((y_pred == y_true))
    confidence /= len(loader)

    try:
        return np.mean(val_loss), perf, acc, confidence, analog_res
    except:
        return np.mean(val_loss), perf, acc, confidence
