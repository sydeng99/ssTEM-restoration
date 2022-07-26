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
import torch.nn.functional as F
from joblib import Parallel
from joblib import delayed
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from utils.flow_synthesis import gen_line, gen_flow
from utils.image_warp import image_warp
from utils.flow_display import dense_flow

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
		self.gt_line = cfg.DATA.gt_line

		# read raw data
		self.train_txt = cfg.DATA.train_txt
		self.interp_train_txt = cfg.DATA.interp_train_txt
		f = open(os.path.join(self.folder_name, self.train_txt), 'r')
		self.train_list = [x[:-1] for x in f.readlines()]
		f.close()
		f = open(os.path.join(self.folder_name, self.interp_train_txt), 'r')
		self.interp_train_list = [x[:-1] for x in f.readlines()]
		f.close()

		self.num = len(self.train_list)
		print('image number: ', self.num)
		self.det_size = 256
		self.offset = (self.crop_size[0] - self.det_size) // 2

	def read_img(self, img_id1, img_id2):
		img_name = img_id1.split(' ')
		# img1 = np.asarray(Image.open(os.path.join(self.folder_name, img_name[0])))
		img2 = np.asarray(Image.open(os.path.join(self.folder_name, img_name[1])))
		# img3 = np.asarray(Image.open(os.path.join(self.folder_name, img_name[2])))
		img_interp = np.asarray(Image.open(os.path.join(self.folder_name, img_id2)))
		return img2, img_interp

	def __getitem__(self, index):
		# s = random.randint(-self.scale_range, self.scale_range)
		s = 0
		crop_size_x = self.crop_size[0] + s
		crop_size_y = self.crop_size[1] + s

		k = random.randint(0, self.num - 1)
		img2, img_interp = self.read_img(self.train_list[k], self.interp_train_list[k])
		img_h, img_w = img2.shape

		i = random.randint(0, img_h - crop_size_x)
		j = random.randint(0, img_w - crop_size_y)
		img2 = img2[i:i+crop_size_x, j:j+crop_size_y]
		img_interp = img_interp[i:i+crop_size_x, j:j+crop_size_y]
		
		img2 = img2[np.newaxis, :, :]
		img_interp = img_interp[np.newaxis, :, :]
		im_lb = np.concatenate([img2, img_interp], axis=0)
		
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
		
		img2 = im_lb[0]
		img2_lb = img2.copy()
		img2_lb = img2_lb[self.offset:-self.offset, self.offset:-self.offset]

		# random brightness, contrast and saturation
		if self.color_jitter:
			img2 = self._color_jitter(img2)
		
		img2, flow = self.degradation(img2)
		# img2 = self.noise(img2)
		if self.gt_line:
			mask_line = np.ones_like(img2)
			mask_line[img2 == 0] = 0
			img2_lb = np.multiply(img2_lb, mask_line)

		img2 = img2[np.newaxis,:,:]
		img2 = np.repeat(img2, 3, 0)
		img_interp = im_lb[1]
		img_interp = img_interp[self.offset:-self.offset, self.offset:-self.offset]
		img_interp = img_interp[np.newaxis,:,:]
		img_interp = np.repeat(img_interp, 3, 0)
		im = np.concatenate([img2, img_interp], axis=0)
		im = im.astype(np.float32) / 255.0

		# lb = np.transpose(flow, (2, 0, 1))
		lb = img2_lb
		lb = lb[np.newaxis, :, :]
		lb = lb.astype(np.float32) / 255.0
		
		if self.gauss_noise:
			im = self._gauss_noise(im)
		
		# elastic transform
		if self.elastic_trans:
			im, lb = self._elastic_transform(im, lb)
		
		return im, lb
	
	def __len__(self):
		return int(sys.maxsize)
	
	def degradation(self, img):
		flag = False
		while flag == False:
			height = self.crop_size[0]
			width = self.crop_size[0]
			line_width = random.randint(5, 20)
			fold_width = random.randint(line_width+1, 80)

			# two end points
			# 1 --> top line (0, x)
			# 2 --> right line (x, width)
			# 3 --> bottom line (height, x)
			# 4 --> left line (x, 0)
			k1 = random.randint(1, 4)
			k2 = random.randint(1, 4)
			while k1 == k2:
				k2 = random.randint(1, 4)
			
			if k1 == 1:
				x = random.randint(1, width-1)
				p1 = [0, x]
			elif k1 == 2:
				x = random.randint(1, height-1)
				p1 = [x, width]
			elif k1 == 3:
				x = random.randint(1, width-1)
				p1 = [height, x]
			else:
				x = random.randint(1, height-1)
				p1 = [x, 0]
			
			if k2 == 1:
				x = random.randint(1, width-1)
				p2 = [0, x]
			elif k2 == 2:
				x = random.randint(1, height-1)
				p2 = [x, width]
			elif k2 == 3:
				x = random.randint(1, width-1)
				p2 = [height, x]
			else:
				x = random.randint(1, height-1)
				p2 = [x, 0]
			
			dis_k = random.uniform(0.00001, 0.1)
			k, b = gen_line(p1, p2)
			flow, flow2, mask = gen_flow(height, width, k, b, line_width, fold_width, dis_k)

			deformed = image_warp(img, flow, mode='bilinear')  # nearest or bilinear

			deformed = (deformed * mask).astype(np.uint8)
			deformed = deformed[self.offset:-self.offset, self.offset:-self.offset]
			flow = flow[self.offset:-self.offset, self.offset:-self.offset]
			flow2 = flow2[self.offset:-self.offset, self.offset:-self.offset]
			# mask = mask[self.offset:-self.offset, self.offset:-self.offset]

			pos = np.where(deformed == 0)
			count = len(pos[0])
			if count < 100:
				flag = False
			else:
				flag = True
		# deformed = deformed + (1 - mask) * random.randint(0, 60)
		# deformed = (deformed).astype(np.uint8)

		return deformed, flow2
	
	def noise(self, img):
		mask = np.ones_like(img)
		mask[img == 0] = 0
		ran_reginal_contrast = random.uniform(0.4, 1.0)
		ran_w = random.randint(50, 200)
		ran_h = random.randint(50, 200)
		ran_pointx = random.randint(0, self.det_size-ran_h)
		ran_pointy = random.randint(0, self.det_size-ran_w)
		contrst_box = img[ran_pointx:ran_pointx+ran_h, ran_pointy:ran_pointy+ran_w]
		contrst_box = ran_reginal_contrast * (contrst_box - np.mean(img)) + np.mean(img)
		img[ran_pointx:ran_pointx+ran_h, ran_pointy:ran_pointy+ran_w] = contrst_box
		img = np.multiply(img, mask)
		return img

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

	cfg_file = 'sff_flowunet.yaml'
	with open('./config/' + cfg_file, 'r') as f:
		cfg = AttrDict( yaml.load(f) )
	
	out_path = os.path.join(cfg.DATA.folder_name, 'temp')
	if not os.path.exists(out_path):
		os.mkdir(out_path)
	data = Train(cfg)
	t = time.time()
	for i in range(0, 20):
		# print('processing ' + str(i))
		im, lb = iter(data).__next__()
		img2 = (im[0:3] * 255).astype(np.uint8)
		img_interp = (im[3:6] * 255).astype(np.uint8)
		img_interp = np.transpose(img_interp, (1,2,0))
		lb = lb.transpose((1,2,0))
		flow_img = dense_flow(lb)
		# import pdb; pdb.set_trace()
		img2 = np.transpose(img2, (1,2,0))
		h, w, c = img2.shape
		img2[0,:] = 0
		img2[:,0] = 0
		img2[h-1,:] = 0
		img2[:,w-1] = 0
		de_img2 = image_warp(img2, lb, mode='nearest')

		# img2_tensor = img2[np.newaxis, :, :, :]
		# img2_tensor = torch.from_numpy(img2_tensor)
		# lb_tensor = lb[np.newaxis, :, :, :]
		# lb_tensor = torch.from_numpy(lb_tensor)
		# de_img2 = F.grid_sample(img2_tensor, lb_tensor, mode='bilinear', padding_mode='zeros')
		# de_img2 = de_img2.numpy().squeeze()
		# de_img2 = np.transpose(de_img2, (1,2,0))
		cat1 = np.concatenate([img2, img_interp], axis=1)
		cat2 = np.concatenate([de_img2, flow_img], axis=1)
		img_cat = np.concatenate([cat1, cat2], axis=0)
		Image.fromarray(img_cat).save(os.path.join(out_path, str(i).zfill(4)+'.png'))
	print(time.time() - t)