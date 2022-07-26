import os
import re
import cv2
import sys
import argparse
import numpy as np
from time import time
import torch
import torch.nn as nn
from networks import UNet, IFNet, FusionNet
from skimage import io
from PIL import Image
import os
import time
from utils.gray2tensor import Gray2Tensor,Tensor2Gray

def TestFusion(model_path, input_data_path, im1,im2_degra, im2_mask,im3_degra, im3_mask,im4, save_path,if_multi_gpu=False):

    print('Building model on ', end='', flush=True)
    #t1 = time()
    device = torch.device('cuda')
    model_denoise=UNet(1,1)
    model_fusion = FusionNet(1, 1)

    model_denoise=model_denoise.to(device)
    model_fusion=model_fusion.to(device)
    model_vfi = IFNet().to(device)

    model_denoise = model_denoise.eval()
    model_fusion=model_fusion.eval()
    model_vfi=model_vfi.eval()

    cuda_count = torch.cuda.device_count()
    if  if_multi_gpu==True:
        print('%d GPUs ... ' % cuda_count, end='', flush=True)
        model_fusion=nn.DataParallel(model_fusion)
        model_denoise=nn.DataParallel(model_denoise)
        model_vfi=nn.DataParallel(model_vfi)
    else:
        print('a single GPU ... ', end='', flush=True)


    print('Resuming weights from %s ... ' % model_path, end='', flush=True)
    checkpoint_vfi = torch.load(model_path+'model_vfi.ckpt')
    model_vfi.load_state_dict(checkpoint_vfi['model_weights'])

    checkpoint_denoise=torch.load(model_path+'model_denoise.ckpt')
    model_denoise.load_state_dict(checkpoint_denoise['model_weights'])

    checkpoint_fusion=torch.load(model_path+'model_fusion.ckpt')
    model_fusion.load_state_dict(checkpoint_fusion['model_weights'])

    model_denoise=model_denoise.cuda()
    model_fusion=model_fusion.cuda()
    model_vfi=model_vfi.cuda()

    print('finish')

    with torch.no_grad():
        im_input1 = np.asarray(Image.open(input_data_path + im1))
        im_input2_degra = np.asarray(Image.open(input_data_path + im2_degra))
        im_input2_degra_mask = np.asarray(Image.open(input_data_path + im2_mask)) 
        im_input3_degra = np.asarray(Image.open(input_data_path + im3_degra))
        im_input3_degra_mask = np.asarray(Image.open(input_data_path + im3_mask)) 
        im_input4 = np.asarray(Image.open(input_data_path + im4))

        if im_input2_degra_mask.ndim == 3:
            im_input2_degra_mask = im_input2_degra_mask[:, :, 0]
        if im_input3_degra_mask.ndim == 3:
            im_input3_degra_mask = im_input3_degra_mask[:, :, 0]


        all_1 = (np.ones_like(im_input2_degra_mask) * 255.).astype('float')
        im_input2_degra_mask_r = (all_1 - im_input2_degra_mask).astype('uint8')
        im_input3_degra_mask_r = (all_1 - im_input3_degra_mask).astype('uint8')

        [h, w] = im_input1.shape
        if h % 32 != 0:
            im_input1 = im_input1[:h - h % 32, :w - w % 32]
            im_input2_degra = im_input2_degra[:h - h % 32, :w - w % 32]
            im_input3_degra = im_input3_degra[:h - h % 32, :w - w % 32]
            im_input4 = im_input4[:h - h % 32, :w - w % 32]

            im_input2_degra_mask = im_input2_degra_mask[:h - h % 32, :w - w % 32]
            im_input2_degra_mask_r = im_input2_degra_mask_r[:h - h % 32, :w - w % 32]
            im_input3_degra_mask = im_input3_degra_mask[:h - h % 32, :w - w % 32]
            im_input3_degra_mask_r = im_input3_degra_mask_r[:h - h % 32, :w - w % 32]

        im_input1 = Gray2Tensor(im_input1)
        im_input2_degra = Gray2Tensor(im_input2_degra)
        im_input3_degra = Gray2Tensor(im_input3_degra)
        im_input4 = Gray2Tensor(im_input4)

        im_input2_degra_mask = Gray2Tensor(im_input2_degra_mask)
        im_input2_degra_mask_r = Gray2Tensor(im_input2_degra_mask_r)
        im_input3_degra_mask = Gray2Tensor(im_input3_degra_mask)
        im_input3_degra_mask_r = Gray2Tensor(im_input3_degra_mask_r)

        print(im_input1.shape)
        print(im_input2_degra.shape)

        print('Start testing')
        start_time = time.time()

        inputs_vfi = torch.cat((im_input1, im_input1, im_input1,
                                im_input4, im_input4, im_input4), 1)
        vfi_pred1 = torch.unsqueeze(model_vfi(inputs_vfi)[:, 0], 1)  # keep the dim in channel
        vfi_pred2 = torch.unsqueeze(model_vfi(inputs_vfi)[:, 1], 1)

        denoise_pred_1 = model_denoise(im_input2_degra)
        denoise_pred_2 = model_denoise(im_input3_degra)

        fusion_input1_1 = torch.mul(vfi_pred1, im_input2_degra_mask_r)
        fusion_input1_2 = torch.mul(denoise_pred_1, im_input2_degra_mask)
        fusion_input2_1 = torch.mul(vfi_pred2, im_input3_degra_mask_r)
        fusion_input2_2 = torch.mul(denoise_pred_2, im_input3_degra_mask)

        fusion_input1 = torch.add(fusion_input1_1, fusion_input1_2)
        fusion_input2 = torch.add(fusion_input2_1, fusion_input2_2)
        pred1 = model_fusion(fusion_input1_1, fusion_input1_2)
        pred2 = model_fusion(fusion_input2_1, fusion_input2_2)

        print("output shape: ", pred1.shape)
        elapsed_time = time.time() - start_time

        vfi_pred1 = Tensor2Gray(vfi_pred1)
        vfi_pred2 = Tensor2Gray(vfi_pred2)
        denoise_pred1 = Tensor2Gray(denoise_pred_1)
        denoise_pred2 = Tensor2Gray(denoise_pred_2)
        pred1 = Tensor2Gray(pred1)
        pred2 = Tensor2Gray(pred2)
        fusion_input1 = Tensor2Gray(fusion_input1)
        fusion_input2 = Tensor2Gray(fusion_input2)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        Image.fromarray(pred1).save(save_path +  'pred1.png')
        Image.fromarray(pred2).save(save_path +  'pred2.png')

        print("It takes {}s for processing.".format(elapsed_time))


if __name__=='__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('-mp', '--model_path')
        parser.add_argument('-dp', '--input_data_path')
        parser.add_argument('-im1', '--img1')
        parser.add_argument('-im2d', '--im2_degra')
        parser.add_argument('-im2m', '--im2_mask')
        parser.add_argument('-im3d', '--im3_degra')
        parser.add_argument('-im3m', '--im3_mask')
        parser.add_argument('-im4', '--img4')
        parser.add_argument('-sp', '--save_path')
        parser.add_argument('-mGPU', '--if_multi_gpu')
        opt = parser.parse_args()

        #def TestFusion(model_path, input_data_path, im1,im2_degra, im2_mask,im3_degra, im3_mask,im4, save_path,if_multi_gpu=False):

        TestFusion(opt.model_path, opt.input_data_path, opt.img1, opt.im2_degra, \
            opt.im2_mask, opt.im3_degra, opt.im3_mask, opt.img4, opt.save_path, opt.if_multi_gpu)