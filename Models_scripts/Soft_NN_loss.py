import torch
import torch.nn as nn
import numpy as np

class Soft_Nearset_Neighbor_Loss():
    def __init__(self, Temp=0.07):
        super(Soft_Nearset_Neighbor_Loss, self).__init__()
        self.temperature = Temp
        self.eps = 1e-6

    def forward(self, input, target):
        loss = 0
        b, n = input.size()
        inx = np.arange(b).tolist()
        for B in range(b):
            cur_indices = inx[:B] + inx[B + 1:]
            cur_indices_tensor = torch.tensor(cur_indices).to(input.device)
            ex_tensor = torch.index_select(input=input, dim=0, index=cur_indices_tensor)
            ex_label = torch.index_select(input=target, dim=0, index=cur_indices_tensor)

            same_label_logic = (ex_label == target[B])

            cur_nom = torch.sum(torch.exp(-(torch.matmul(input=torch.unsqueeze(input[B, :], dim=0), other=ex_tensor[same_label_logic, :].t()) / (torch.norm(torch.unsqueeze(input[B, :], dim=0)) * torch.norm(ex_tensor[same_label_logic, :].t()) + self.eps)) / self.temperature))
            cur_dom = torch.sum(torch.exp(-(torch.matmul(input=torch.unsqueeze(input[B, :], dim=0), other=ex_tensor.t()) / (torch.norm(torch.unsqueeze(input[B, :], dim=0)) * torch.norm(ex_tensor.t()) + self.eps)) / self.temperature))
            cur_log = torch.log(cur_nom / cur_dom + self.eps)

            loss += cur_log

        loss *= -(1 / b)

        return loss
