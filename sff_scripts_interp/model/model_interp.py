import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable, gradcheck
from libs.sepconv.SeparableConvolution import SeparableConvolution
# import sepconv


class IFNet(nn.Module):
	def __init__(self, kernel_size=51):
		super(IFNet, self).__init__()
		conv_kernel = (3, 3)
		conv_stride = (1, 1)
		conv_padding = 1
		sep_kernel = kernel_size # OUTPUT_1D_KERNEL_SIZE

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
		self.upconv51_1 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu)
		self.upconv51_2 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu)
		self.upconv51_3 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu)
		self.upconv51_4 = self._kernel_module(64, sep_kernel, conv_kernel, conv_stride, conv_padding, self.upsamp, self.relu)
		
		upscale_factor = 2
		self.srconv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
		self.srconv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
		self.srconv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
		self.srconv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
		self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
		
		self.pad = nn.ReplicationPad2d(sep_kernel // 2)
		self.separable_conv = SeparableConvolution.apply

		# self.separable_conv = sepconv.FunctionSepconv()
		#self._check_gradients(self.separable_conv)

		self.apply(self._weight_init)
		self.sigmoid = nn.Sigmoid()

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
		k2h = self.upconv51_1(x)
		k2v = self.upconv51_2(x)
		k1h = self.upconv51_3(x)
		k1v = self.upconv51_4(x)
		padded_i2 = self.pad(i2)
		padded_i1 = self.pad(i1)

		# ------------ Local convolutions ------------
		y = self.separable_conv(padded_i2, k2v, k2h) + self.separable_conv(padded_i1, k1v, k1h)
		# y = sepconv.FunctionSepconv(padded_i2, k2v, k2h) + sepconv.FunctionSepconv(padded_i1, k1v, k1h)
		# y = torch.mean(y, dim=1, keepdim=True)
		output = torch.mean(y, dim=1, keepdim=True)
		# output = self.sigmoid(output)
		
		# ------------ Super-Resolution ------------
		# y_bil = self.upsamp(y)
		# res = self.relu(self.srconv1(y))
		# res = self.relu(self.srconv2(res))
		# res = self.relu(self.srconv3(res))
		# res = self.pixel_shuffle(self.srconv4(res))
		# output = y_bil + res
		return output

	@staticmethod
	def _check_gradients(func):
		print('Starting gradient check...')
		sep_kernel = 51 # OUTPUT_1D_KERNEL_SIZE
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
			#init.kaiming_normal_(m.weight, 0, 'fan_in', 'relu')


if __name__ == '__main__':
	import cv2
	import time
	import numpy as np
	model = IFNet()#.to('cuda:0')
	
	input = np.random.random((1, 6, 256, 256)).astype(np.float32)
	#i1 = cv2.resize(i1, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
	#i2 = cv2.resize(i2, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
	#input = np.expand_dims(np.concatenate([np.expand_dims(i1, axis=0), np.expand_dims(i2, axis=0)], axis=0), axis=0)
	
	x = torch.tensor(input)#.to('cuda:0')
	t1 = time.time()
	out = model(x)
	
	print(out.shape)
	print('COST TIME:', (time.time() - t1))
