from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import cv2
import h5py
import torch
import time
import numpy as np
import random
import torchvision
from PIL import Image
import tifffile
import multiprocessing
from joblib import Parallel
from joblib import delayed
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


class Train(Dataset):
	def __init__(self, cfg):
		super(Train, self).__init__()
		# multiprocess settings
		num_cores = multiprocessing.cpu_count()
		self.parallel = Parallel(n_jobs=num_cores, backend='threading')
		self.use_mp = False
		self.cfg = cfg

		# basic settings
		self.folder_name = cfg.DATA.folder_name
		self.crop_size = list(cfg.DATA.patch_size)
		self.invalid_border = cfg.DATA.invalid_border
		self.scale_range = cfg.DATA.scale_range
		
		# simple augmentations
		self.random_fliplr = cfg.DATA.AUG.random_fliplr
		self.random_flipud = cfg.DATA.AUG.random_flipud
		self.random_flipz = cfg.DATA.AUG.random_flipz
		self.random_rotation = cfg.DATA.AUG.random_rotation
		
		# color augmentations
		self.color_jitter = cfg.DATA.AUG.color_jitter
		self.brightness = cfg.DATA.AUG.COLOR.brightness
		self.contrast = cfg.DATA.AUG.COLOR.contrast
		self.saturation = cfg.DATA.AUG.COLOR.saturation
		
		# gauss noise
		self.gauss_noise = cfg.DATA.AUG.gauss_noise
		self.gauss_mean = cfg.DATA.AUG.GAUSS.gauss_mean
		self.gauss_sigma = cfg.DATA.AUG.GAUSS.gauss_sigma

		# elastic transform
		self.elastic_trans = cfg.DATA.AUG.elastic_trans
		self.alpha_range = cfg.DATA.AUG.ELASTIC.alpha_range
		self.sigma = cfg.DATA.AUG.ELASTIC.sigma
		self.shave = cfg.DATA.AUG.ELASTIC.shave

		# Normalization
		# self.normalization = cfg.DATA.AUG.normalization
		# self.mean = cfg.DATA.AUG.NORM.mean
		# self.std = cfg.DATA.AUG.NORM.std
		
		# extend crop size
		self.crop_size[0] = self.crop_size[0] + 2 * self.shave if self.elastic_trans else self.crop_size[0]
		self.crop_size[1] = self.crop_size[1] + 2 * self.shave if self.elastic_trans else self.crop_size[1]
		
		# color jitter
		self.cj = torchvision.transforms.ColorJitter(self.brightness, self.contrast, self.saturation, hue=0)

		# interpolation swap
		self.swap = cfg.DATA.AUG.swap

		# read raw data
		self.train_txt = cfg.DATA.train_txt
		f = open(os.path.join(self.folder_name, self.train_txt), 'r')
		self.train_list = [x[:-1] for x in f.readlines()]

		self.num = len(self.train_list)
		print('image number: ', self.num)

	def read_img(self, img_id):
		img_name = img_id.split(' ')
		img1 = np.asarray(Image.open(os.path.join(self.folder_name, img_name[0])))
		img2 = np.asarray(Image.open(os.path.join(self.folder_name, img_name[1])))
		img3 = np.asarray(Image.open(os.path.join(self.folder_name, img_name[2])))
		return img1, img2, img3

	def __getitem__(self, index):
		# s = random.randint(-self.scale_range, self.scale_range)
		s = 0
		crop_size_x = self.crop_size[0] + s
		crop_size_y = self.crop_size[1] + s

		k = random.randint(0, self.num - 1)
		img1, img2, img3 = self.read_img(self.train_list[k])
		img_h, img_w = img1.shape

		i = random.randint(0, img_h - crop_size_x)
		j = random.randint(0, img_w - crop_size_y)
		img1 = img1[i:i+crop_size_x, j:j+crop_size_y]
		img2 = img2[i:i+crop_size_x, j:j+crop_size_y]
		img3 = img3[i:i+crop_size_x, j:j+crop_size_y]
		
		img1 = img1[np.newaxis, :, :]
		img2 = img2[np.newaxis, :, :]
		img3 = img3[np.newaxis, :, :]
		im_lb = np.concatenate([img1, img2, img3], axis=0)
		
		# random flip
		if self.random_fliplr and random.uniform(0, 1) < 0.5:
			for j in range(im_lb.shape[0]): im_lb[j, :, :] = np.fliplr(im_lb[j, :, :])
		if self.random_flipud and random.uniform(0, 1) < 0.5:
			for j in range(im_lb.shape[0]): im_lb[j, :, :] = np.flipud(im_lb[j, :, :])
		if self.random_flipz and random.uniform(0, 1) < 0.5:
			for j in range(im_lb.shape[0]): im_lb[j, :, :] = np.transpose(im_lb[j, :, :])
		
		# random rotation
		if self.random_rotation:
			r = random.randint(0, 3)
			for j in range(im_lb.shape[0]): im_lb[j, :, :] = np.rot90(im_lb[j, :, :], r)
		
		if self.swap and random.uniform(0, 1) < 0.5:
			tmp = im_lb[2].copy()
			im_lb[2] = im_lb[0]
			im_lb[0] = tmp
		
		img1 = im_lb[0:1]
		img2 = im_lb[1:2]
		img3 = im_lb[2:3]

		img1 = np.repeat(img1, 3, 0)
		img3 = np.repeat(img3, 3, 0)
		im = np.concatenate([img1, img3], axis=0)
		im = im.astype(np.float32) / 255.0
		lb = img2
		lb = lb.astype(np.float32) / 255.0
		
		# random brightness, contrast and saturation
		if self.color_jitter:
			im = self._color_jitter(im)
		
		if self.gauss_noise:
			im = self._gauss_noise(im)
		
		# elastic transform
		if self.elastic_trans:
			im, lb = self._elastic_transform(im, lb)
		
		return im, lb
	
	def __len__(self):
		return int(sys.maxsize)
	
	@staticmethod
	def _shave(im, border):
		if len(im.shape) == 4:
			return im[:, :, border[0] : -border[0], border[1] : -border[1]]
		elif len(im.shape) == 3:
			return im[:, border[0] : -border[0], border[1] : -border[1]]
		elif len(im.shape) == 2:
			return im[border[0] : -border[0], border[1] : -border[1]]
		else:
			raise NotImplementedError
	
	@staticmethod
	def _pad_with(vector, pad_width, iaxis, kwargs):
		pad_value = kwargs.get('padder', 10)
		vector[:pad_width[0]] = pad_value
		vector[-pad_width[1]:] = pad_value
		return vector
	
	def _fliplr(self, im_lb):
		outputs = []
		for sub_vol in im_lb:
			results = []
			for input in sub_vol:
				results.append(np.expand_dims(np.fliplr(input), axis=0))
			outputs.append(np.expand_dims(np.concatenate(results, axis=0), axis=0))
		return np.concatenate(outputs, axis=0)
	
	def _flipud(self, im_lb):
		outputs = []
		for sub_vol in im_lb:
			results = []
			for input in sub_vol:
				results.append(np.expand_dims(np.flipud(input), axis=0))
			outputs.append(np.expand_dims(np.concatenate(results, axis=0), axis=0))
		return np.concatenate(outputs, axis=0)
	
	def _rotate(self, im_lb, r):
		outputs = []
		for sub_vol in im_lb:
			results = []
			for input in sub_vol:
				results.append(np.expand_dims(np.rot90(input, r), axis=0))
			outputs.append(np.expand_dims(np.concatenate(results, axis=0), axis=0))
		return np.concatenate(outputs, axis=0)
	
	def _color_jitter(self, im):
		# im = im.transpose([1,2,0])
		new_im = np.asarray(self.cj(Image.fromarray(im)))
		# new_im = new_im.transpose([2,0,1])
		return new_im
		# return np.asarray(self.cj(Image.fromarray(im)))
		# results = []
		# for input in im:
		# 	results.append(np.expand_dims(np.asarray(self.cj(Image.fromarray(input))), axis=0))
		# return np.concatenate(results, axis=0)
	
	@staticmethod
	def _draw_grid(im, grid_size=50, gray_level=255):
		for i in range(0, im.shape[1], grid_size):
			cv2.line(im, (i, 0), (i, im.shape[0]), color=(gray_level,))
		for j in range(0, im.shape[0], grid_size):
			cv2.line(im, (0, j), (im.shape[1], j), color=(gray_level,))
	
	@staticmethod
	def _map(input, indices, shape):
		return np.expand_dims(map_coordinates(input, indices, order=1).reshape(shape), axis=0)
	
	def _gauss_noise(self, img):
		img = img.astype(np.float32) / 255.0
		noise = np.random.normal(self.gauss_mean, self.gauss_sigma ** 0.5, img.shape)
		out = img + noise
		if out.min() < 0:
			low_clip = -1.
		else:
			low_clip = 0.
		out = np.clip(out, low_clip, 1.0)
		out = (out * 255).astype(np.uint8)
		return out
	
	def _elastic_transform(self, image_in, label_in, random_state=None):
		"""Elastic deformation of image_ins as described in [Simard2003]_.
		.. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
		   Convolutional Neural Networks applied to Visual Document Analysis", in
		   Proc. of the International Conference on Document Analysis and
		   Recognition, 2003.
		"""
		alpha = np.random.uniform(0, self.alpha_range)
		
		if random_state is None:
			random_state = np.random.RandomState(None)
		
		shape = image_in.shape[1:]
		# shape = image_in.shape
		
		dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma, mode='constant', cval=0) * alpha
		dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma, mode='constant', cval=0) * alpha
		
		x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
		indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
		
		if self.use_mp:
			image_out = np.concatenate(self.parallel(delayed(self._map)(input, indices, shape) for input in image_in), axis=0)
		else:
			# image_out = map_coordinates(image_in, indices, order=1).reshape(shape)
			image_out = []
			for input in image_in:
				image_out.append(np.expand_dims(map_coordinates(input, indices, order=1).reshape(shape), axis=0))
			image_out = np.concatenate(image_out, axis=0)
		
		if self.use_mp:
			label_out = []
			for sub_vol in label_in:
				results = np.concatenate(self.parallel(delayed(self._map)(input, indices, shape) for input in sub_vol), axis=0)
				label_out.append(np.expand_dims(results, axis=0))
			label_out = np.concatenate(label_out, axis=0)
		else:
			label_out = map_coordinates(label_in, indices, order=1).reshape(shape)
			# label_out = []
			# for sub_vol in label_in:
			# 	results = []
			# 	for input in sub_vol:
			# 		results.append(np.expand_dims(map_coordinates(input, indices, order=1).reshape(shape), axis=0))
			# 	label_out.append(np.expand_dims(np.concatenate(results, axis=0), axis=0))
			# label_out = np.concatenate(label_out, axis=0)
		
		image_out = self._shave(image_out, [self.shave, self.shave])
		label_out = self._shave(label_out, [self.shave, self.shave])
		
		return image_out, label_out

class Provider(object):
	def __init__(self, stage, cfg):
			#patch_size, batch_size, num_workers, is_cuda=True):
		self.stage = stage
		if self.stage == 'train':
			self.data = Train(cfg)
			self.batch_size = cfg.TRAIN.batch_size
			self.num_workers = cfg.TRAIN.num_workers
		elif self.stage == 'valid':
			# return valid(folder_name, kwargs['data_list'])
			pass
		else:
			raise AttributeError('Stage must be train/valid')
		self.is_cuda = cfg.TRAIN.is_cuda
		self.data_iter = None
		self.iteration = 0
		self.epoch = 1
	
	def __len__(self):
		return self.data.num_per_epoch
	
	def build(self):
		if self.stage == 'train':
			self.data_iter = iter(DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                                             shuffle=False, drop_last=False, pin_memory=True))
		else:
			self.data_iter = iter(DataLoader(dataset=self.data, batch_size=1, num_workers=0,
                                             shuffle=False, drop_last=False, pin_memory=True))
	
	def next(self):
		if self.data_iter is None:
			self.build()
		try:
			batch = self.data_iter.next()
			self.iteration += 1
			if self.is_cuda:
				batch[0] = batch[0].cuda()
				batch[1] = batch[1].cuda()
			return batch[0], batch[1]
		except StopIteration:
			self.epoch += 1
			self.build()
			self.iteration += 1
			batch = self.data_iter.next()
			if self.is_cuda:
				batch[0] = batch[0].cuda()
				batch[1] = batch[1].cuda()
			return batch[0], batch[1]


if __name__ == '__main__':
	import yaml
	from attrdict import AttrDict
	""""""

	cfg_file = 'ms_adloss.yaml'
	with open('./config/' + cfg_file, 'r') as f:
		cfg = AttrDict( yaml.load(f) )
	
	out_path = os.path.join(cfg.DATA.folder_name, 'temp')
	if not os.path.exists(out_path):
		os.mkdir(out_path)
	data = Train(cfg)
	t = time.time()
	for i in range(0, 20):
		im, lb = iter(data).__next__()
		img1 = (im[0] * 255).astype(np.uint8)
		img3 = (im[3] * 255).astype(np.uint8)
		img2 = (lb[0] * 255).astype(np.uint8)
		img_cat = np.concatenate([img1, img2, img3], axis=1)
		Image.fromarray(img_cat).save(os.path.join(out_path, str(i).zfill(4)+'.png'))
	print(time.time() - t)