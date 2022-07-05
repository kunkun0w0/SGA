import torch
import torch.nn as nn
import torch.nn.functional as F
from spectral_normalization import SpectralNorm


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        return self.main(x) + x


class ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, spec_norm=False, LR=0.2, stride=1, up=False):
        super(ConvBlock, self).__init__()

        self.up = up
        if self.up:
            self.up_sample = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.up_sample = None

        if spec_norm:
            self.main = nn.Sequential(
                SpectralNorm(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride, padding=1)),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=False)
            )

        else:
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=False),
            )

    def forward(self, x1, x2=None):
        if self.up_sample is not None:
            x1 = self.up_sample(x1)
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x = torch.cat([x2, x1], dim=1)
            return self.main(x)
        else:
            return self.main(x1)


class Gconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Gconv, self).__init__()
        self.src_fc = nn.Linear(in_ch, out_ch)
        self.msg_fc = nn.Linear(in_ch, out_ch)
        self.bn = nn.BatchNorm1d(out_ch, affine=True, track_running_stats=True)

    def forward(self, A, source, message):
        src = self.src_fc(source)
        msg = self.msg_fc(message)

        gen = torch.bmm(A, F.leaky_relu(msg, negative_slope=0.2)) + F.leaky_relu(src, negative_slope=0.2)

        return self.bn(gen.permute(0, 2, 1)).permute(0, 2, 1)
