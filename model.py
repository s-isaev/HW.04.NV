import torch
from config import ProcessConfig
import torch.nn.functional as F
import torch.nn as nn

class ResSubblock(torch.nn.Module):
    def __init__(self, config: ProcessConfig):
        super(ResSubblock, self).__init__()

        self.len_res_subblock = config.len_res_subblock

        self.layers = nn.ModuleList()
        for i in range(self.len_res_subblock):
            self.layers.append(nn.Conv1d(config.hidden, config.hidden, 5, padding='same'))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.leaky_relu(x)
        return x

class ResBlock(torch.nn.Module):
    def __init__(self, config: ProcessConfig):
        super(ResBlock, self).__init__()

        self.n_res_subblocks = config.n_res_subblocks

        self.subblocks = nn.ModuleList()
        for i in range(self.n_res_subblocks):
            self.subblocks.append(ResSubblock(config))

    def forward(self, x):
        for subblock in self.subblocks:
            x_old = x
            x = subblock(x)
            x = (x + x_old) / 2
        return x


class UpsampleBlock(torch.nn.Module):
    def __init__(self, config: ProcessConfig):
        super(UpsampleBlock, self).__init__()

        self.n_res_blocks = config.n_res_blocks
        self.device = config.device

        self.convt = nn.ConvTranspose1d(
            in_channels=config.hidden,
            out_channels=config.hidden,
            kernel_size=11,
            padding=5,
            stride=2,
            output_padding=1
        )
        self.resblocks = nn.ModuleList()
        for i in range(self.n_res_blocks):
            self.resblocks.append(ResBlock(config))

    def forward(self, x):
        x = F.leaky_relu(self.convt(x))
        x_res = torch.zeros(x.shape).to(self.device)
        for resblock in self.resblocks:
            x_res = x_res + resblock(x)/self.n_res_blocks
        return x_res

class Generator(torch.nn.Module):
    def __init__(self, config: ProcessConfig):
        super(Generator, self).__init__()

        self.welcome = nn.Conv1d(80, config.hidden, 1)
        self.upsamples = nn.ModuleList()
        target = 256
        while target != 1:
            self.upsamples.append(UpsampleBlock(config))
            target //= 2
        self.final = nn.Conv1d(config.hidden, 1, 1)
        

    def forward(self, x):
        x = F.leaky_relu(self.welcome(x))
        for upsample in self.upsamples:
            x = upsample(x)
        x = self.final(x).squeeze(1)
        return torch.tanh(x)