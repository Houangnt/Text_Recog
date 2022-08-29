import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tps_transform import GridGenerator

__all__ = ['spin_w_tps']


class SPINwTPS(nn.Module):
    def __init__(self, img_r_size, nc=1, k=6, num_fiducial=20, device=torch.device('cuda')):
        super(SPINwTPS, self).__init__()
        self.img_r_size = img_r_size
        self.num_fiducial = num_fiducial
        self.nc = nc
        self.k = k
        self.device = device

        self.shared_conv, self.spn, self.ain, self.sigmoid, self.pool = get_spn_ain(nc, 2 * (self.k + num_fiducial) + 2)
        self.instance_norm = nn.InstanceNorm2d(nc)
        self.beta = self.get_beta()

        self.grid_generator = GridGenerator(self.num_fiducial, self.img_r_size, self.device)
        self.initialize_weights()

    def initialize_weights(self):
        for module in [self.shared_conv, self.spn, self.ain]:
            for m in module.modules():
                t = type(m)
                if t is nn.Conv2d:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif t is nn.BatchNorm2d:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                    m.eps = 1e-3
                    m.momentum = 0.03
                elif t is nn.Linear:
                    m.weight.data.normal_(0, 0.001)
                    m.bias.data.zero_()
        self.spn.Linear_2.weight.data.fill_(0)
        init = np.array([.0] * self.k + [5.] + [.0] * self.k + [-5.00])

        ctrl_pts_x = np.linspace(-1.0, 1.0, int(self.num_fiducial / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(self.num_fiducial / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(self.num_fiducial / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        initial_bias = initial_bias.reshape(-1)
        init = np.concatenate([init, initial_bias], axis=0)
        self.spn.Linear_2.bias.data = torch.from_numpy(init).float().view(-1)

    def get_beta(self):
        b1 = []
        b2 = []
        for i in range(1, self.k + 1):
            tmp = round(math.log(1 - 0.5 * i / (self.k + 1)) / math.log(0.5 * i / (self.k + 1)), 2)
            b1.append(tmp)
            b2.append(round(1 / tmp, 2))
        return b1 + [1.] + b2

    def forward(self, x):
        x = x.to(device=self.device)
        feature = self.shared_conv(x)
        sp_output = self.spn(feature).view(x.size(0), 2 * (self.k + self.num_fiducial) + 2, 1)

        alpha = sp_output[:, 2 * self.k + 1, 0]
        alpha = self.sigmoid(alpha).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        w_sp = sp_output[:, : 2 * self.k + 1, :]
        offsets = self.pool(self.ain(feature))
        offsets = self.sigmoid(offsets)
        offsets = F.interpolate(offsets, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        x = .5 * (x + 1) * (1 - alpha) + offsets * alpha
        w_s = w_sp.unsqueeze(-1).unsqueeze(-1)
        powered = torch.stack([x.pow(p) for p in self.beta], dim=1)

        x = (powered * w_s).sum(dim=1)
        x = self.sigmoid(self.instance_norm(x)) * 2 + 1

        batch_c_prime = sp_output[:, 2 * self.k + 2:, :].view(x.shape[0], self.num_fiducial, 2)
        build_p_prime = self.grid_generator.build_P_prime(batch_c_prime).reshape(
            [-1, self.img_r_size[0], self.img_r_size[1], 2])
        x = F.grid_sample(x, build_p_prime, padding_mode='border', align_corners=True)

        return x


def get_spn_ain(input_channel, n_output):
    shared_conv = nn.Sequential(
        nn.Conv2d(input_channel, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(32), nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(64), nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(128), nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
    spn_output = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(256), nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(256), nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(512), nn.ReLU(True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1, -1),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256), nn.ReLU(True)
    )
    spn_output.add_module('Linear_2', nn.Linear(256, n_output))
    ain_output = nn.Sequential(
        nn.Conv2d(128, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(True),
        nn.Conv2d(16, input_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    sigmoid = nn.Sigmoid()
    pool = nn.MaxPool2d(2, 2)
    return shared_conv, spn_output, ain_output, sigmoid, pool


def spin_w_tps(opt):
    n_fiducial = opt.transformation.num_fiducial
    i_r_size = opt.transformation.i_r_size_H, opt.transformation.i_r_size_W
    return SPINwTPS(i_r_size, opt.nc, n_fiducial, opt.F, opt.device)


if __name__ == '__main__':
    example = torch.rand(3, 3, 32, 100)
    model = SPINwTPS((32, 100), 3, 6, 20, torch.device('cpu'))
    model(example)
