import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from natten import NeighborhoodAttention1D
import torch.nn.functional as F
from pthflops import count_ops
from natten.flops import get_flops


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        x = x[:, :, :-self.chomp_size].contiguous()
        return x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, local=14):
        super(TemporalBlock, self).__init__()
        self.k = kernel_size
        # print("local", local)
        self.dilation = dilation
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout1d(dropout)
        # print(local + (dilation*(self.k-1)))
        self.conv2 = NeighborhoodAttention1D(dim=n_outputs, kernel_size=kernel_size,
                                             dilation=dilation, num_heads=2)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout1d(dropout)
        self.net_1 = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1)
        self.net_2 = nn.Sequential(self.conv2)
        self.net_3 = nn.Sequential(self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1)
        self.relu = nn.ReLU()
        nn.init.kaiming_normal_(self.conv1.weight)

    def forward(self, x):

        out = self.net_1(x)
        out = F.pad(out, (self.conv2.dilation*(self.k-1), 0))
        out = out.permute(0, 2, 1)
        out = self.net_2(out)
        out = out.permute(0, 2, 1)
        out = self.net_3(out)
        res = self.downsample(x)
        return self.relu1(out + res)


class TemporalConvNet(nn.Module):

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, dilation_size=2):

        super(TemporalConvNet, self).__init__()
        print("NTCNN")
        layers = []
        self.num_channels = num_channels
        num_levels = len(num_channels)
        self.kernel_size = kernel_size
        self.dilation = dilation_size
        for i in range(num_levels):
            dilation_size = self.dilation ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout, local=14)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def receptive_field(model):
    return sum([model.dilation**(l-1)*(model.kernel_size-1) for l in range(len(model.num_channels), 0, -1)]) + 1
