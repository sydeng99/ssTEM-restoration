import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable, gradcheck
from libs.sepconv.SeparableConvolution import SeparableConvolution
import torch.nn.functional as F
from skimage import morphology

class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        conv_kernel = (3, 3)
        conv_stride = (1, 1)
        conv_padding = 1
        sep_kernel = 51  # OUTPUT_1D_KERNEL_SIZE

        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.upsamp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.relu = nn.ReLU(inplace=False)

        self.conv32 = self._conv_module(6, 32, conv_kernel, conv_stride, conv_padding, self.relu)
        self.conv64 = self._conv_module(32, 64, conv_kernel, conv_stride, conv_padding, self.relu)
        self.conv128 = self._conv_module(64, 128, conv_kernel, conv_stride, conv_padding, self.relu)
        self.conv256 = self._conv_module(128, 256, conv_kernel, conv_stride, conv_padding, self.relu)
        self.conv512 = self._conv_module(256, 512, conv_kernel, conv_stride, conv_padding, self.relu)
        self.conv512x512 = self._conv_module(512, 512, conv_kernel, conv_stride, conv_padding, self.relu)
        self.upsamp512 = self._upsample_module(512, 512, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu)
        self.upconv256 = self._conv_module(512, 256, conv_kernel, conv_stride, conv_padding, self.relu)
        self.upsamp256 = self._upsample_module(256, 256, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu)
        self.upconv128 = self._conv_module(256, 128, conv_kernel, conv_stride, conv_padding, self.relu)
        self.upsamp128 = self._upsample_module(128, 128, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu)
        self.upconv64 = self._conv_module(128, 64, conv_kernel, conv_stride, conv_padding, self.relu)
        self.upsamp64 = self._upsample_module(64, 64, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu)
        self.upconv51_11 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp,
                                               self.relu)
        self.upconv51_12 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp,
                                               self.relu)
        self.upconv51_13 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp,
                                               self.relu)
        self.upconv51_14 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp,
                                               self.relu)
        self.upconv51_15 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp,
                                               self.relu)
        self.upconv51_16 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp,
                                               self.relu)
        self.upconv51_17 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp,
                                               self.relu)
        self.upconv51_18 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp,
                                               self.relu)

        self.upconv51_21 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp,
                                               self.relu)
        self.upconv51_22 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp,
                                               self.relu)
        self.upconv51_23 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp,
                                               self.relu)
        self.upconv51_24 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp,
                                               self.relu)
        self.upconv51_25 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp,
                                               self.relu)
        self.upconv51_26 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp,
                                               self.relu)
        self.upconv51_27 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp,
                                               self.relu)
        self.upconv51_28 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp,
                                               self.relu)


        self.pad = nn.ReplicationPad2d(sep_kernel // 2)
        self.separable_conv = SeparableConvolution.apply

        self.apply(self._weight_init)

    def forward(self, x):
        i1 = x[:, :3]
        i2 = x[:, 3:6]

        # ------------ Contraction ------------
        x = self.conv32(x)
        x = self.pool(x)
        x64 = self.conv64(x)
        x128 = self.pool(x64)
        x128 = self.conv128(x128)
        x256 = self.pool(x128)
        x256 = self.conv256(x256)
        x512 = self.pool(x256)
        x512 = self.conv512(x512)
        x = self.pool(x512)
        x = self.conv512x512(x)

        # ------------ Expansion ------------
        x = self.upsamp512(x)
        x += x512
        x = self.upconv256(x)
        x = self.upsamp256(x)
        x += x256
        x = self.upconv128(x)
        x = self.upsamp128(x)
        x += x128
        x = self.upconv64(x)
        x = self.upsamp64(x)
        x += x64

        # ------------ Final branches ------------
        k11h = self.upconv51_11(x)
        k11v = self.upconv51_12(x)
        k12h = self.upconv51_13(x)
        k12v = self.upconv51_14(x)
        

        k21h = self.upconv51_21(x)
        k21v = self.upconv51_22(x)
        k22h = self.upconv51_23(x)
        k22v = self.upconv51_24(x)
        
        padded_i2 = self.pad(i2)
        padded_i1 = self.pad(i1)

        # ------------ Local convolutions ------------
        y1 = self.separable_conv(padded_i2, k12v, k12h) + self.separable_conv(padded_i1, k11v, k11h)
        y1 = torch.mean(y1, dim=1, keepdim=True)
        y2 = self.separable_conv(padded_i2, k22v, k22h) + self.separable_conv(padded_i1, k21v, k21h)
        y2 = torch.mean(y2, dim=1, keepdim=True)


        y = torch.cat((y1, y2), 1)
        channel=2

        return y

    @staticmethod
    def _check_gradients(func):
        print('Starting gradient check...')
        sep_kernel = 51  # OUTPUT_1D_KERNEL_SIZE
        inputs = (
            Variable(torch.randn(2, 3, sep_kernel, sep_kernel).cuda(), requires_grad=False),
            Variable(torch.randn(2, sep_kernel, 1, 1).cuda(), requires_grad=True),
            Variable(torch.randn(2, sep_kernel, 1, 1).cuda(), requires_grad=True),
        )
        test = gradcheck(func, inputs, eps=1e-2, atol=1e-2, rtol=1e-2)
        print('Gradient check result:', test)

    @staticmethod
    def _conv_module(in_channels, out_channels, kernel, stride, padding, relu):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, kernel, stride, padding), relu,
            torch.nn.Conv2d(in_channels, in_channels, kernel, stride, padding), relu,
            torch.nn.Conv2d(in_channels, out_channels, kernel, stride, padding), relu,
        )

    @staticmethod
    def _kernel_module(in_channels, out_channels, kernel, stride, padding, upsample, relu):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, kernel, stride, padding), relu,
            torch.nn.Conv2d(in_channels, in_channels, kernel, stride, padding), relu,
            torch.nn.Conv2d(in_channels, out_channels, kernel, stride, padding), relu,
            upsample,
            torch.nn.Conv2d(out_channels, out_channels, kernel, stride, padding)
        )

    @staticmethod
    def _upsample_module(in_channels, out_channels, kernel, stride, padding, upsample, relu):
        return torch.nn.Sequential(
            upsample, torch.nn.Conv2d(in_channels, out_channels, kernel, stride, padding), relu,
        )

    @staticmethod
    def _weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.orthogonal_(m.weight, init.calculate_gain('relu'))

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class FusionNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(FusionNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x_in1, x_in2):
        x=torch.add(x_in1, x_in2)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits