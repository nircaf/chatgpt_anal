import torch

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms


class NeuroBNet(nn.Module):
    def __init__(self, in_ch_cnn=1, in_ch_linear=1, n_classes=3, mat_size=224, num_of_layers=4, cnn_groth_factor=8, use_cuda=True):
        super(NeuroBNet, self).__init__()
        # CNN
        self.num_of_layers = num_of_layers
        self.cnn2dense_len = int(self.clac_cnn2dense(mat_size=mat_size, num_of_cnn_layer=num_of_layers, num_of_last_cnn_ch=(num_of_layers * cnn_groth_factor * in_ch_cnn)))
        self.cnn_layers = []
        self.linear_layers = []
        self.bn_layers = []
        for layer in range(num_of_layers):
            if use_cuda:
                if layer == 0:
                    self.cnn_layers.append(nn.Conv2d(in_channels=in_ch_cnn, out_channels=cnn_groth_factor * in_ch_cnn, kernel_size=3, padding=1).cuda())
                    self.linear_layers.append(nn.Linear(in_ch_linear, cnn_groth_factor * in_ch_cnn).cuda())
                    self.bn_layers.append(nn.BatchNorm2d(num_features=cnn_groth_factor * in_ch_cnn).cuda())
                else:
                    self.cnn_layers.append(
                        nn.Conv2d(in_channels=(layer * cnn_groth_factor * in_ch_cnn), out_channels=((layer + 1) * cnn_groth_factor * in_ch_cnn), kernel_size=3, padding=1).cuda())
                    self.linear_layers.append(nn.Linear((layer * cnn_groth_factor * in_ch_cnn), ((layer + 1) * cnn_groth_factor * in_ch_cnn)).cuda())
                    self.bn_layers.append(nn.BatchNorm2d(num_features=((layer + 1) * cnn_groth_factor * in_ch_cnn)).cuda())
            else:
                if layer == 0:
                    self.cnn_layers.append(nn.Conv2d(in_channels=in_ch_cnn, out_channels=cnn_groth_factor * in_ch_cnn, kernel_size=3, padding=1))
                    self.linear_layers.append(nn.Linear(in_ch_linear, cnn_groth_factor * in_ch_cnn))
                    self.bn_layers.append(nn.BatchNorm2d(num_features=cnn_groth_factor * in_ch_cnn))
                else:
                    self.cnn_layers.append(
                        nn.Conv2d(in_channels=(layer * cnn_groth_factor * in_ch_cnn), out_channels=((layer + 1) * cnn_groth_factor * in_ch_cnn), kernel_size=3, padding=1))
                    self.linear_layers.append(nn.Linear((layer * cnn_groth_factor * in_ch_cnn), ((layer + 1) * cnn_groth_factor * in_ch_cnn)))
                    self.bn_layers.append(nn.BatchNorm2d(num_features=((layer + 1) * cnn_groth_factor * in_ch_cnn)))


        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.dropout2d = nn.Dropout2d(0.25)

        # End of Net
        # self.end_fc1 = nn.Linear(self.cnn2dense_len + (num_of_layers * cnn_groth_factor * in_ch_cnn), 120)
        self.end_fc1 = nn.Linear(self.cnn2dense_len, self.cnn2dense_len // 128)
        self.end_fc2 = nn.Linear(self.cnn2dense_len // 128, self.cnn2dense_len // 512)
        self.end_fc3 = nn.Linear(self.cnn2dense_len // 512, n_classes)

    def clac_cnn2dense(self, mat_size=224, num_of_cnn_layer=4, num_of_last_cnn_ch=1, max_pool_size=2):
        if isinstance(mat_size, list) and isinstance(max_pool_size, list):
            length = (((mat_size[0] * mat_size[1]) / ((max_pool_size[0] * max_pool_size[1]) ** num_of_cnn_layer)) * num_of_last_cnn_ch)
        elif (isinstance(mat_size, list)) and  (not isinstance(max_pool_size, list)):
            length = (((mat_size[0] * mat_size[1]) / (
                        (max_pool_size * max_pool_size) ** num_of_cnn_layer)) * num_of_last_cnn_ch)
        elif (not isinstance(mat_size, list)) and (isinstance(max_pool_size, list)):
            length = (((mat_size * mat_size) / (
                    (max_pool_size[0] * max_pool_size[1]) ** num_of_cnn_layer)) * num_of_last_cnn_ch)
        elif (not isinstance(mat_size, list)) and (not isinstance(max_pool_size, list)):
            length = (((mat_size * mat_size) / (
                    (max_pool_size * max_pool_size) ** num_of_cnn_layer)) * num_of_last_cnn_ch)

        return length

    def convolve_cnn_linear_output_layers(self, x1, x2):
        output = x1 * x2[:, :, None, None]

        return output

    def forward(self, image, data):
        for layer in range(self.num_of_layers):
            if layer == 0:
                x1 = image
                x2 = data

            x1 = self.cnn_layers[layer](x1)
            x2 = self.linear_layers[layer](x2)
            x1 = self.convolve_cnn_linear_output_layers(x1=x1, x2=x2)
            # x1 = self.dropout2d(x1)
            x1 = self.bn_layers[layer](x1)
            x1 = self.relu(x1)
            x1 = self.pool(x1)


        x1_dense = torch.squeeze(x1.view(x1.shape[0], -1, self.cnn2dense_len), dim=1)
        # x_dense = torch.cat((x1_dense, x2), dim=1)
        x_dense = self.end_fc1(x1_dense)
        # x_dense = self.dropout(x_dense)
        x_dense = self.end_fc2(x_dense)
        # x_dense = self.dropout(x_dense)
        result = self.end_fc3(x_dense)

        return result


class NeuroBNet_old(nn.Module):
    def __init__(self, in_ch_cnn=1, in_ch_linear=1, n_classes=3, mat_size=224, num_of_layers=4, cnn_groth_factor=8):
        super(NeuroBNet_old, self).__init__()
        # CNN
        self.num_of_layers = num_of_layers
        self.cnn2dense_len = int(
            self.clac_cnn2dense(mat_size=mat_size, num_of_cnn_layer=num_of_layers, num_of_last_cnn_ch=(64 * in_ch_cnn)))
        self.conv1 = nn.Conv2d(in_channels=in_ch_cnn, out_channels=8*in_ch_cnn, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8 * in_ch_cnn, out_channels=16 * in_ch_cnn, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16 * in_ch_cnn, out_channels=32 * in_ch_cnn, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32 * in_ch_cnn, out_channels=64 * in_ch_cnn, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features=8 * in_ch_cnn)
        self.bn2 = nn.BatchNorm2d(num_features=16 * in_ch_cnn)
        self.bn3 = nn.BatchNorm2d(num_features=32 * in_ch_cnn)
        self.bn4 = nn.BatchNorm2d(num_features=64 * in_ch_cnn)

        # Linear
        self.fc1 = nn.Linear(in_ch_linear, 8 * in_ch_cnn)
        self.fc2 = nn.Linear(8 * in_ch_cnn, 16 * in_ch_cnn)
        self.fc3 = nn.Linear(16 * in_ch_cnn, 32 * in_ch_cnn)
        self.fc4 = nn.Linear(32 * in_ch_cnn, 64 * in_ch_cnn)

        # End of Net
        self.end_fc1 = nn.Linear(self.cnn2dense_len, 120)
        self.end_fc2 = nn.Linear(120, 84)
        self.end_fc3 = nn.Linear(84, n_classes)

    def clac_cnn2dense(self, mat_size=224, num_of_cnn_layer=4, num_of_last_cnn_ch=1):
        length = ((mat_size / (2 ** num_of_cnn_layer)) * num_of_last_cnn_ch)
        return length

    def convolve_cnn_linear_output_layers(self, x1, x2):
        output = x1.deepcopy()
        for ch_inx in range(len(x2)):
            output[:, ch_inx, :, :] = x1[:, ch_inx, :, :] * x2[:, ch_inx]

        return output

    def forward(self, image, data):
        # First layer
        x1 = self.conv1(image)
        x2 = self.fc1(data)
        x1 = self.convolve_cnn_linear_output_layers(x1=x1, x2=x2)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.pool(x1)

        # Second layer
        x1 = self.conv2(x1)
        x2 = self.fc2(x2)
        x1 = self.convolve_cnn_linear_output_layers(x1=x1, x2=x2)
        x1 = self.bn2(x1)
        x1 = self.relu(x1)
        x1 = self.pool(x1)

        # Third layer
        x1 = self.conv3(x1)
        x2 = self.fc3(x2)
        x1 = self.convolve_cnn_linear_output_layers(x1=x1, x2=x2)
        x1 = self.bn3(x1)
        x1 = self.relu(x1)
        x1 = self.pool(x1)

        # Forth layer
        x1 = self.conv4(x1)
        x2 = self.fc4(x2)
        x1 = self.convolve_cnn_linear_output_layers(x1=x1, x2=x2)
        x1 = self.bn4(x1)
        x1 = self.relu(x1)
        x1 = self.pool(x1)

        x1_dense = x1.view(x1.shape[0], -1, self.cnn2dense_len)
        x1_dense = self.end_fc1(x1_dense)
        x1_dense = self.end_fc2(x1_dense)
        result = self.end_fc3(x1_dense)

        return result
