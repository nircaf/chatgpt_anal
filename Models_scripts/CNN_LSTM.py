import torch
import math
import numpy as np
import scipy.io as sci
import torch.nn as nn
import os
import h5py
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


use_cuda = torch.cuda.is_available()

N, T, D, L, O = 100, 10, 200, 8, 2  # batch_size, seq_length , word_dim	,leads
hidden_size = 50
eps = 1e-6

###############################################################################
# class TrainDataset(Data.Dataset):
#     def __init__(self):
#         a = np.array(h5py.File('/home/lu/code/pytorch/data_dir/traindata.mat')['traindata'])
#         # b=np.array(h5py.File('/home/lu/code/pytorch/data_dir/data11.mat')['train_data1'])
#         self.data = np.transpose(a)
#
#     def __getitem__(self, idx):
#         data_ori = torch.from_numpy(self.data[idx])
#         data = data_ori[0:16000]
#         label = data_ori[16000]
#         return data, label
#
#     def __len__(self):
#         return len(self.data)
#
#
# trainset = TrainDataset()
# train_loader = Data.DataLoader(trainset, batch_size=N, shuffle=True)


################################################################################
# class TestDataset(Data.Dataset):
#     def __init__(self):
#         a = np.array(h5py.File('/home/lu/code/pytorch/data_dir/testdata.mat')['testdata'])
#         # b=np.array(h5py.File('/home/lu/code/pytorch/data_dir/data11.mat')['test_data1'])
#         self.data = np.transpose(a)
#
#     def __getitem__(self, idx):
#         data_ori = torch.from_numpy(self.data[idx])
#         data = data_ori[0:16000]
#         label = data_ori[16000]
#         return data, label
#
#     def __len__(self):
#         return len(self.data)
#
#
# testset = TestDataset()
# test_loader = Data.DataLoader(testset, batch_size=1, shuffle=True)


##################################################
class CNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.001, max_length=L, args=None):
        super(CNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.dropout = nn.Dropout(self.dropout_p)
        self.batch_size = args.batch_size
        self.first_cnn_layer_out = 4
        self.second_cnn_layer_out = 8
        self.third_cnn_layer_out = 16

        self.first_cnn_kernel_length = args.first_cnn_kernel_length
        self.second_cnn_kernel_length = args.second_cnn_kernel_length
        self.third_cnn_kernel_length = args.third_cnn_kernel_length

        # self.length_after_cnn = int((args.raw_time_duration - \
        #                             (self.first_cnn_kernel_length + self.second_cnn_kernel_length + \
        #                              self.third_cnn_kernel_length)) / 2) + 1

        self.length_after_cnn = args.raw_time_duration

        self.conv1 = nn.Conv2d(1, self.first_cnn_layer_out, (1, self.first_cnn_kernel_length), padding=(0, self.first_cnn_kernel_length // 2))
        self.conv2 = nn.Conv2d(self.first_cnn_layer_out, self.second_cnn_layer_out, (1, self.second_cnn_kernel_length), padding=(0, self.second_cnn_kernel_length // 2))
        self.conv3 = nn.Conv2d(self.second_cnn_layer_out, self.third_cnn_layer_out, (1, self.third_cnn_kernel_length), padding=(0, self.third_cnn_kernel_length // 2))
        self.max_pool_2D = nn.MaxPool2d((1, 3), stride=(1, 2))


        # self.conv4 = nn.Conv2d(3,1,10)

        self.fc1 = nn.Linear(self.batch_size * self.third_cnn_layer_out * self.length_after_cnn, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, encoder_outputs):
        # print(encoder_outputs)
        cnn_out = F.relu(self.conv1(encoder_outputs))
        cnn_out = F.relu(self.conv2(cnn_out))
        cnn_out = F.relu(self.conv3(cnn_out))
        # cnn_out = F.max_pool1d(F.sigmoid(self.conv4(cnn_out)),2)
        # print(cnn_out)
        cnn_out = cnn_out.view(-1, self.batch_size * self.third_cnn_layer_out * self.length_after_cnn)
        output = F.relu(self.fc1(cnn_out))
        # output = self.out(output)

        return output


class RNN(nn.Module):
    def __init__(self, input_size, hidden_szie1, hidden_szie2, output_size, args=None):
        super(RNN, self).__init__()

        self.T = args.RNN_seq_len
        if args.RNN_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_szie1, 1, dropout=0.3)
            self.r2h = nn.Linear(hidden_szie1, hidden_szie2)
            self.h2o = nn.Linear(hidden_szie2, output_size)
        elif args.RNN_type == 'Conv3D':
            self.rnn = nn.Conv3d(input_size, hidden_szie1, 1, dropout=0.3)

    def forward(self, input):
        hidden, _ = self.rnn(input)
        fc1 = F.relu(self.r2h(hidden[self.T - 1]))
        output = self.h2o(fc1)
        # output = F.softmax(output)

        return output

class Coordinator(nn.Module):
    def __init__(self, args=None):
        super(Coordinator, self).__init__()
        self.hidden_size = args.num_of_electrodes * args.n_classes
        self.output_size = args.n_classes
        self.fc1 = nn.Linear(self.hidden_size, self.output_size)
        self.num_of_electrodes = args.num_of_electrodes

    def forward(self, input, mode='dense'):
        if mode == 'dense':
            output = torch.softmax(self.fc1(input))
        elif mode =='majority':
            tmp = input.view(input.shape[0], self.num_of_electrodes, -1)
            Argmax = torch.softmax(tmp, dim=2)
            output = torch.zeros(Argmax.shape[0], 1)
            for jj in range(Argmax.shape[0]):
                output[jj] = torch.softmax(torch.bincount(Argmax[jj, :])).item()
        # output = F.softmax(output)

        return output

def train_validation(input_variable, target_variable, cnn, cnn_optimizer, rnn, rnn_optimizer, coord, coord_optimizer, criterion, args=None, mode='train'):

    if args is not None:
        N = args.batch_size
        T = args.RNN_seq_len
        D = args.RNN_input_dim
    else:
        pass
    rnns_outputs = []
    cnn_optimizer.zero_grad()
    rnn_optimizer.zero_grad()
    coord_optimizer.zero_grad()


    cnn_output = cnn(torch.transpose(input_variable, 2, 3))
    for ii in range(cnn_output.shape[0]):
        cnn_output_cur = cnn_output[ii, :].view(N, T, D).transpose(0, 1)
        rnn_output = rnn(cnn_output_cur)
        rnns_outputs.append(rnn_output)

    coord_input = torch.stack(rnns_outputs, dim=1)
    coord_input = coord_input.view(coord_input.shape[0], -1)
    coord_output = coord(coord_input, mode=args.coord_mode)

    if args.coord_mode == 'dense':
        loss = criterion(coord_output, target_variable)
    elif args.coord_mode == 'majority':
        loss = torch.nn.functional.mse_loss(coord_output.to(target_variable.device), target_variable)

    if mode == 'train':
        loss.backward()

        cnn_optimizer.step()
        rnn_optimizer.step()
        coord_optimizer.step()
    else:
        pass

    return loss.data, coord_output


def test(input_variable, cnn, rnn, coord, args=None):
    rnns_outputs = []
    if args is not None:
        N = args.batch_size
        T = args.RNN_seq_len
        D = args.RNN_input_dim
    else:
        pass
    cnn_output = cnn(torch.transpose(input_variable, 2, 3))
    for ii in range(cnn_output.shape[0]):
        cnn_output_cur = cnn_output[ii, :].view(N, T, D).transpose(0, 1)
        rnn_output = rnn(cnn_output_cur)
        rnns_outputs.append(rnn_output)


    coord_input = torch.stack(rnns_outputs, dim=1)
    coord_input = coord_input.view(coord_input.shape[0], -1)
    coord_output = coord(coord_input, mode=args.coord_mode)
    top_n, top_i = coord_output.data.topk(1)

    return top_i[0][0], coord_output


def trainIters(cnn, rnn, coord, epoch, learning_rate=0.001, args=None, loaders=None):
    n_epochs = epoch
    current_loss_train = 0
    current_loss_val = 0
    all_losses_train = []
    all_losses_val = []
    err_rate = []
    confusion_val = torch.zeros(args.n_classes, args.n_classes)
    confusion_test = torch.zeros(args.n_classes, args.n_classes)
    learning_rate = args.lr_rate
    err = 0

    cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    rnn_optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    coord_optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    train_loader = loaders['train_loader']
    validation_loader = loaders['val_loader']
    test_loader = loaders['test_loader']

    results = torch.zeros(n_epochs, 3, max(len(train_loader), len(validation_loader), len(test_loader)), args.n_classes)

    for epoch in range(1, n_epochs + 1):
        ### Train
        if args is not None:
            N = args.batch_size
            L = args.raw_time_duration
        else:
            pass

        cnn.train()
        rnn.train()
        coord.train()
        for step1, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = Variable(batch_x.type('torch.cuda.FloatTensor'))
            batch_y = Variable(batch_y.type('torch.cuda.LongTensor'))
            try:
                loss_train, output = train_validation(batch_x.view(N, 1, L, -1), batch_y, cnn, cnn_optimizer, rnn, rnn_optimizer, coord, coord_optimizer, criterion, args=args, mode='train')
            except:
                try:
                    loss_train, output = train_validation(batch_x.view(N, 1, L, -1), batch_y, cnn, cnn_optimizer, rnn, rnn_optimizer, criterion)
                except:
                    continue
            current_loss_train += loss_train
            results[epoch, 0, step1, :] = output[0, :]

        ### Validation
        if args is not None:
            N = args.batch_size
            L = args.raw_time_duration
        else:
            pass
        cnn.eval()
        rnn.eval()
        coord.eval()
        for step2, (val_x, val_y) in enumerate(validation_loader):
            val_x = Variable(val_x.type('torch.cuda.FloatTensor'))
            val_y = val_y.type('torch.cuda.LongTensor')
            try:
                loss_val, output = train_validation(val_x.view(N, 1, L, -1), val_y, cnn, cnn_optimizer, rnn, rnn_optimizer, coord, coord_optimizer, criterion, args=args, mode='val')
            except:
                try:
                    loss_val, output = train_validation(val_x.view(N, 1, L, -1), val_y, cnn, cnn_optimizer, rnn, rnn_optimizer, criterion)
                except:
                    continue
            current_loss_val += loss_val
            results[epoch, 1, step2, :] = output[0, :]
            try:
                guess, coord_output = test(val_x.view(N, 1, L, -1), cnn, rnn, coord, args=args)
                confusion_val[guess][val_y[0]] += 1
            except:
                continue
        tmp_acc = 0
        tmp_sen = 0
        for kk in range(args.n_classes):
            tmp_sen += (confusion_val[kk][kk]) / (torch.sum(confusion_val[kk][:])+eps)
            tmp_acc += confusion_val[kk][kk]
        sen = tmp_sen / args.n_classes
        acc = tmp_acc / step2

        all_losses_train.append(current_loss_train / step1)
        all_losses_val.append(current_loss_val / step2)
        err_rate.append(acc * 100)

        current_loss = 0
        print('%d epoch: acc = %.2f%%, sen = %.2f%%, loss_train = %.2f, loss_val = %.2f' % (epoch, acc * 100, sen * 100, (current_loss_train / step1), (current_loss_val / step2)))
        err = 0
        confusion_val[:] = 0
        current_loss_train = 0
        current_loss_val = 0
        loss_train = 0
        loss_val = 0

    ### Test
    if args is not None:
        N = args.batch_size
        L = args.raw_time_duration
    else:
        pass
    for step3, (test_x, test_y) in enumerate(test_loader):
        test_x = Variable(test_x.type('torch.cuda.FloatTensor'))
        test_y = test_y.type('torch.cuda.LongTensor')
        try:
            guess, coord_output = test(test_x.view(N, 1, L, -1), cnn, rnn, coord, args=args)
            confusion_test[guess][test_y[0]] += 1
        except:
            continue


    fig, ax = plt.subplots()
    ax.plot(range(1, n_epochs + 1), all_losses_train, 'k--', label='Train Loss')
    ax.plot(range(1, n_epochs + 1), all_losses_val, 'k:', label='Validation Loss')
    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
    plt.title('loss')

    plt.figure()
    plt.plot(err_rate)
    plt.title('err')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_test.numpy())
    fig.colorbar(cax)

    plt.show()


    return results
