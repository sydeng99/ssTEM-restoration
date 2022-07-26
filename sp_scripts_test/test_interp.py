import os
import re
import sys
import argparse
import logging
import numpy as np
from time import time
import time
from datetime import datetime
from PIL import Image

from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from networks import IFNet




def TestVFI(model_path, input_data_path, input_img1, input_img2, save_path, mGPU=False):
    print('Building model on ', end='', flush=True)
    
    if mGPU==True:
        device = torch.device('cuda')
        model_vfi = IFNet().to(device)
        model_vfi=model_vfi.eval()
        cuda_count = torch.cuda.device_count()
        print('%d GPUs ... ' % cuda_count, end='', flush=True)
        model_vfi=nn.DataParallel(model_vfi)
    else:
        model_vfi = IFNet().cuda()
        print('a single GPU ... ', end='', flush=True)
        

    print('Resuming weights from %s ... ' % model_path, end='', flush=True)
    checkpoint_denoise=torch.load(model_path)
    model_vfi.load_state_dict(checkpoint_denoise['model_weights'])

    model_vfi=model_vfi.cuda()

    with torch.no_grad():
        data_path=input_data_path
        ims=os.listdir(data_path)

        im_input0 = np.asarray(Image.open(data_path +input_img1))
        im_input3 = np.asarray(Image.open(data_path + input_img2))

        im_input0 = (im_input0 / 255.).astype('float32')
        im_input3 = (im_input3 / 255.).astype('float32')

        im_input0=im_input0[np.newaxis,np.newaxis,:,:,]
        im_input0=Variable(torch.from_numpy(im_input0))
        im_input3=im_input3[np.newaxis,np.newaxis,:,:,]
        im_input3=Variable(torch.from_numpy(im_input3))
        im_input0=im_input0.cuda()
        im_input3=im_input3.cuda()

        print('Start testing')
        start_time = time.time()
        print("input shape: ", im_input0.shape)
        [b,c,h,w]=im_input0.shape
        if h%4!=0:
            im_input0=im_input0[:,:,:h-h%4,:w-w%4]
            im_input3=im_input3[:,:,:h-h%4,:w-w%4]
        print("input shape: ", im_input0.shape)

        inputs_vfi = torch.cat((im_input0, im_input0, im_input0, \
                                im_input3, im_input3, im_input3), 1)
        vfi_pred1 = torch.unsqueeze(model_vfi(inputs_vfi)[:, 0], 1)  # keep the dim in channel
        vfi_pred2 = torch.unsqueeze(model_vfi(inputs_vfi)[:, 1], 1)

        print("output shape: ", vfi_pred1.shape)
        elapsed_time = time.time() - start_time

        im_output=vfi_pred1
        im_output = im_output.cpu()
        print("output.shape(cpu):", im_output.shape)
        [b, c, w, h] = im_output.shape
        im_output = im_output.data.numpy()
        im_out = np.zeros(shape=[w, h], dtype='uint8')
        im_out = im_output[0, 0, :, :]*255.
        im_out = im_out.astype('uint8')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Image.fromarray(im_out).save(save_path + 'vfi_1.png')

        im_output=vfi_pred2
        im_output = im_output.cpu()
        print("output.shape(cpu):", im_output.shape)
        [b, c, w, h] = im_output.shape
        im_output = im_output.data.numpy()
        im_out = np.zeros(shape=[w, h], dtype='uint8')
        im_out = im_output[0, 0, :, :]*255.
        im_out = im_out.astype('uint8')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Image.fromarray(im_out).save(save_path + 'vfi_2.png')

        print("It takes {}s for processing.".format(elapsed_time))

if __name__=='__main__':
    # def TestVFI(model_path,epoch, input_data_path, input_img1, input_img2, save_path, mGPU)
    parser = argparse.ArgumentParser()
    parser.add_argument('-mp', '--model_path')
    parser.add_argument('-dp', '--input_data_path')
    parser.add_argument('-im1', '--input_img1')
    parser.add_argument('-im2', '--input_img2')
    parser.add_argument('-sp', '--save_path')
    parser.add_argument('-mGPU', '--if_multi_gpu')
    opt = parser.parse_args()

    TestVFI(opt.model_path, opt.input_data_path, opt.input_img1, opt.input_img2, opt.save_path, opt.if_multi_gpu)