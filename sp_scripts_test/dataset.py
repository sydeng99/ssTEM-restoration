from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import math
import numpy as np
import random
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from utils.util import setup_seed

class ImageDataset(Dataset):
    def __init__(self,opts):
        super(ImageDataset).__init__()
        self.data_folder=opts.data_folder
        self.data_txt=opts.data_txt
        transform=[transforms.ToTensor()]
        f_img_path = open(self.data_txt, 'r')
        self.path_list = [x[:-1] for x in f_img_path.readlines()]
        self.patch_size=opts.patch_size
        self.dataset_size=len(self.path_list)
        self.if_rotate=opts.if_rotate
        self.cuda=opts.cuda
        self.if_bdadjust=opts.if_bdadjust
        self.transforms=torchvision.transforms.Compose(transform)
        self.if_use_vfiImg = opts.if_use_vfiImg

    def __getitem__(self, index):
        txt_line=self.path_list[index]
        img_name = txt_line.split(' ')

        # setup_seed(self.random_seed)

        imgs = []
        img_1 = np.asarray(Image.open(self.data_folder + img_name[0]))
        img_2 = np.asarray(Image.open(self.data_folder + img_name[1]))
        img_2_degra = np.asarray(Image.open(self.data_folder + img_name[2]))
        img_3 = np.asarray(Image.open(self.data_folder + img_name[3]))
        img_3_degra = np.asarray(Image.open(self.data_folder + img_name[4]))
        img_4 = np.asarray(Image.open(self.data_folder + img_name[5]))
        
        img_2_degraB1_mask_gradall= np.asarray(Image.open(self.data_folder + img_name[6]))
        img_3_degraB1_mask_gradall= np.asarray(Image.open(self.data_folder + img_name[7]))
        img_2_degraB1_mask_gradall_r = (np.ones_like(img_2_degraB1_mask_gradall)*255-img_2_degraB1_mask_gradall).astype(img_2_degraB1_mask_gradall.dtype)
        img_3_degraB1_mask_gradall_r = (np.ones_like(img_3_degraB1_mask_gradall)*255-img_3_degraB1_mask_gradall).astype(img_3_degraB1_mask_gradall.dtype)


        img_2_degraB1_GenGradMask = np.asarray(Image.open(self.data_folder + img_name[8]))
        img_3_degraB1_GenGradMask = np.asarray(Image.open(self.data_folder + img_name[9]))
        img_2_degraB1_GenGradMask_r = (np.ones_like(img_2_degraB1_GenGradMask)*255-img_2_degraB1_GenGradMask).astype(img_2_degraB1_GenGradMask.dtype)
        img_3_degraB1_GenGradMask_r = (np.ones_like(img_3_degraB1_GenGradMask)*255-img_3_degraB1_GenGradMask).astype(img_3_degraB1_GenGradMask.dtype)


        imgs.append(img_1)                             # 0
        imgs.append(img_2)                             # 1
        imgs.append(img_2_degra)                       # 2
        imgs.append(img_3)                             # 3
        imgs.append(img_3_degra)                       # 4
        imgs.append(img_4)                             # 5

        imgs.append(img_2_degraB1_mask_gradall)        # 6
        imgs.append(img_3_degraB1_mask_gradall)        # 7
        imgs.append(img_2_degraB1_mask_gradall_r)      # 8
        imgs.append(img_3_degraB1_mask_gradall_r)      # 9
        imgs.append(img_2_degraB1_GenGradMask)         # 10
        imgs.append(img_3_degraB1_GenGradMask)         # 11
        imgs.append(img_2_degraB1_GenGradMask_r)       # 12
        imgs.append(img_3_degraB1_GenGradMask_r)       # 13

        if self.if_use_vfiImg:
            img_2_degraB1_vfi = np.asarray(Image.open(self.data_folder + img_name[10]))
            img_3_degraB1_vfi = np.asarray(Image.open(self.data_folder + img_name[11]))

            imgs.append(img_2_degraB1_vfi)                 # 14
            imgs.append(img_3_degraB1_vfi)                 # 15

        imgs_crop = self.CropImg(imgs)

        imgs_rotate = []
        if self.if_rotate:
            a = random.randint(0, 7)
            for jj in range(len(imgs_crop)):
                imgs_rotate.append(self.RotationFlip(imgs_crop[jj], a))
        else:
            imgs_rotate = imgs_crop


        imgs_tensor = []
        for jj in range(len(imgs_rotate)):
            if jj == 2 or jj == 4:  # noisy image, ColorJitter
                imgs_tensor.append(self.load_img(imgs_rotate[jj], True))
            else:
                imgs_tensor.append(self.load_img(imgs_rotate[jj], False))

        imgs_tensor_cuda = []
        if self.cuda:
            for kk in range(len(imgs_tensor)):
                imgs_tensor_cuda.append(imgs_tensor[kk].cuda())
        else:
            imgs_tensor_cuda = imgs_tensor

        if self.if_use_vfiImg:
            return {'img_1': imgs_tensor_cuda[0], 
                    'img_2': imgs_tensor_cuda[1], 
                    'img_2_degra': imgs_tensor_cuda[2],
                    'img_3': imgs_tensor_cuda[3],
                    'img_3_degra': imgs_tensor_cuda[4],
                    'img_4': imgs_tensor_cuda[5],
                    'img_2_degraB1_mask_gradall': imgs_tensor_cuda[6],
                    'img_3_degraB1_mask_gradall': imgs_tensor_cuda[7],
                    'img_2_degraB1_mask_gradall_r': imgs_tensor_cuda[8],
                    'img_3_degraB1_mask_gradall_r': imgs_tensor_cuda[9],
                    'img_2_degraB1_GenGradMask': imgs_tensor_cuda[10],
                    'img_3_degraB1_GenGradMask': imgs_tensor_cuda[11],
                    'img_2_degraB1_GenGradMask_r': imgs_tensor_cuda[12],
                    'img_3_degraB1_GenGradMask_r': imgs_tensor_cuda[13],
                    'img_2_degraB1_vfi': imgs_tensor_cuda[14],
                    'img_3_degraB1_vfi': imgs_tensor_cuda[15]
                    }
        else:
            return {'img_1': imgs_tensor_cuda[0], 
                    'img_2': imgs_tensor_cuda[1], 
                    'img_2_degra': imgs_tensor_cuda[2],
                    'img_3': imgs_tensor_cuda[3],
                    'img_3_degra': imgs_tensor_cuda[4],
                    'img_4': imgs_tensor_cuda[5],
                    'img_2_degraB1_mask_gradall': imgs_tensor_cuda[6],
                    'img_3_degraB1_mask_gradall': imgs_tensor_cuda[7],
                    'img_2_degraB1_mask_gradall_r': imgs_tensor_cuda[8],
                    'img_3_degraB1_mask_gradall_r': imgs_tensor_cuda[9],
                    'img_2_degraB1_GenGradMask': imgs_tensor_cuda[10],
                    'img_3_degraB1_GenGradMask': imgs_tensor_cuda[11],
                    'img_2_degraB1_GenGradMask_r': imgs_tensor_cuda[12],
                    'img_3_degraB1_GenGradMask_r': imgs_tensor_cuda[13]
                    }



    def load_img(self,img,degra_flag):
        img_pil=Image.fromarray(img)
        if self.if_bdadjust and degra_flag:
            if random.random()>0.7:
                img_pil=self.BCAdjust(img_pil)
        tensor_img=self.transforms(img_pil)
        return tensor_img

    def BCAdjust(self,img_pil):
        ad=transforms.ColorJitter(0.2,0.2,0.2)
        img_pil_out=ad(img_pil)
        return img_pil_out

    def CropImg(self,images):
        img=[]
        if images[0].ndim==3:
            [h,w,c]=images[0].shape
        else:
            [h,w]=images[0].shape
        ran_h=random.randint(0,h-self.patch_size)
        ran_w=random.randint(0,w-self.patch_size)
        for ii in range(len(images)):
            img.append(images[ii][ran_h:ran_h+self.patch_size,ran_w:ran_w+self.patch_size])
        return img

    def ShiftCropImg(self,imgs,shift_distance,shift_direction):
        out_imgs=[]
        direction=random.uniform(shift_direction[0],shift_direction[1])
        distance=random.uniform(shift_distance[0],shift_distance[1])
        shift_x=int(math.sin(direction)*distance)
        shift_y=int(math.cos(direction)*distance)
        imgs_1=imgs[:5]
        imgs_2=imgs[5:]
        delta_x=int(math.fabs(shift_x))
        delta_y=int(math.fabs(shift_y))
        if imgs[0].ndim==3:
            [h,w,c]=imgs[0].shape
        else:
            [h,w]=imgs[0].shape
        if shift_x>=0 and shift_y>=0:
            ran_h = random.randint(0, h - self.patch_size - delta_y)
            ran_w = random.randint(0, w - self.patch_size - delta_x)
            for ii in range(len(imgs_1)):
                out_imgs.append(imgs_1[ii][ran_h:ran_h+self.patch_size,ran_w:ran_w+self.patch_size])
            for jj in range(len(imgs_2)):
                out_imgs.append(imgs_2[jj][ran_h+shift_y:ran_h+shift_y+self.patch_size,ran_w+shift_x:ran_w+shift_x+self.patch_size])
        elif shift_x<0 and shift_y>=0:
            ran_h=random.randint(0,h-self.patch_size-delta_y)
            ran_w=random.randint(delta_x,w-self.patch_size)
            for ii in range(len(imgs_1)):
                out_imgs.append(imgs_1[ii][ran_h:ran_h+self.patch_size,ran_w:ran_w+self.patch_size])
            for jj in range(len(imgs_2)):
                out_imgs.append(imgs_2[jj][ran_h+shift_y:ran_h+shift_y+self.patch_size,ran_w+shift_x:ran_w+shift_x+self.patch_size])
        elif shift_x<0 and shift_y<0:
            ran_h=random.randint(delta_y,h-self.patch_size)
            ran_w=random.randint(delta_x,w-self.patch_size)
            for ii in range(len(imgs_1)):
                out_imgs.append(imgs_1[ii][ran_h:ran_h+self.patch_size,ran_w:ran_w+self.patch_size])
            for jj in range(len(imgs_2)):
                out_imgs.append(imgs_2[jj][ran_h+shift_y:ran_h+shift_y+self.patch_size,ran_w+shift_x:ran_w+shift_x+self.patch_size])
        else:
            ran_h=random.randint(delta_y,h-self.patch_size)
            ran_w=random.randint(0,w-self.patch_size-delta_x)
            for ii in range(len(imgs_1)):
                out_imgs.append(imgs_1[ii][ran_h:ran_h+self.patch_size,ran_w:ran_w+self.patch_size])
            for jj in range(len(imgs_2)):
                out_imgs.append(imgs_2[jj][ran_h+shift_y:ran_h+shift_y+self.patch_size,ran_w+shift_x:ran_w+shift_x+self.patch_size])
        return out_imgs

    def RotationFlip(self, image, case):
        if case == 0:
            return image
        elif case == 1:  # flipped
            return np.fliplr(image)
        elif case == 2:  # rotation 90
            return np.rot90(image)
        elif case == 3:  # rotation 90 & flipped
            a = np.rot90(image, 1)
            return np.fliplr(a)
        elif case == 4:  # rotation 180
            return np.rot90(image, 2)
        elif case == 5:  # rotation 180 &flipped
            a = np.rot90(image, 2)
            return np.fliplr(a)
        elif case == 6:  # rotation 270
            return np.rot90(image, 3)
        elif case == 7:  # rotation 270 & flipped
            a = np.rot90(image, 3)
            return np.fliplr(a)

    def DilateMask(self, image, thickness):
        B_img_np = np.asarray(image)
        kernel = np.ones((thickness, thickness), np.uint8)
        B_img_np_dilate = cv2.dilate(B_img_np, kernel, iterations=1)
        mask_d = B_img_np_dilate
        return mask_d

    def ErodeMask(self, image, thickness):
        B_img_np = np.asarray(image)
        kernel = np.ones((thickness, thickness), np.uint8)
        B_img_np_erode = cv2.erode(B_img_np, kernel, iterations=1)
        mask_e = B_img_np_erode
        return mask_e

    def __len__(self):
        return self.dataset_size

class Provider(object):
    def __init__(self, stage, cfg):
        self.stage = stage
        if self.stage == 'train':
            self.data = ImageDataset(cfg)
            self.batch_size = cfg.TRAIN.batch_size
            self.num_workers = cfg.TRAIN.num_workers
        elif self.stage == 'valid':
            # return valid(folder_name, kwargs['data_list'])
            pass
        else:
            raise AttributeError('Stage must be train/valid')
        self.is_cuda = cfg.TRAIN.if_cuda
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        return len(self.data)

    def build(self):
        if self.stage == 'train':
            self.data_iter = iter(
                DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
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
                for k in batch.keys():
                    batch[k]=batch[k].cuda()
            return batch
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = self.data_iter.next()
            if self.is_cuda:
                for k in batch.keys():
                    batch[k]=batch[k].cuda()
            return batch

def tensor2im(input_image, imtype=np.uint8, if_type='+-1'):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  #
        if image_numpy.ndim==2:
            image_numpy=image_numpy[np.newaxis,:,:]
        print(image_numpy.shape)
        if if_type=='+-1':
            image_numpy = (np.transpose(image_numpy,
                                        (1, 2, 0)) + 1) / 2.0 * 255.0
        elif if_type=='01':
            image_numpy = (np.transpose(image_numpy, (1,2,0)))*255.0
        else:
            raise NotImplementedError
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return np.clip(image_numpy[:,:,0], 0, 255).astype(imtype)


def Gray2Tensor(im):
    im=(im / 255.).astype('float32')
    im = im[np.newaxis, np.newaxis, :, :, ]
    tensor = Variable(torch.from_numpy(im))
    tensor=tensor.cuda()
    return tensor

def Tensor2Gray(tensor):
    tensor = tensor.cpu()
    [b, c, w, h] = tensor.shape
    tensor = tensor.data.numpy()
    im_out = tensor[0, 0, :, :] * 255.
    im_out = im_out.astype('uint8')
    return im_out

