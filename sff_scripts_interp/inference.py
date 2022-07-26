# Written by Wei Huang
# 2020/04/06

import os
import cv2
import time
import yaml
import h5py
import argparse
import numpy as np 
from PIL import Image
from attrdict import AttrDict

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from collections import OrderedDict

from model.model_interp import IFNet
from utils.psnr_ssim import compute_psnr, compute_ssim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='ms_l1loss_decay')
    parser.add_argument('-id', '--model_id', type=str, default='interp')
    parser.add_argument('-m', '--mode', type=str, default='valid')  # valid, test
    parser.add_argument('-ip', '--input_path', type=str, default='../data/test/test_cremia/')  # ../data/test/cremia
    parser.add_argument('-t', '--txt_file', type=str, default='cremia_25sff')     # cremia_25sff
    parser.add_argument('-op', '--output_path', type=str, default='../results/cremia/')  # ../results/cremia/
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f))
    
    f_txt = open(args.input_path+ args.txt_file+'.txt', 'r')
    img_list = [int(x[:-1]) for x in f_txt.readlines()]
    f_txt.close()
    img_path = os.path.join(args.input_path, args.txt_file)
    output_path = os.path.join(args.output_path, args.txt_file+'_'+args.model_id)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Build model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = IFNet(kernel_size=cfg.TRAIN.kernel_size).to(device)
    ckpt_path = os.path.join('../trained_models', args.model_id, args.model_id+'.ckpt')
    checkpoint = torch.load(ckpt_path)

    new_state_dict = OrderedDict()
    state_dict = checkpoint['model_weights']
    for k, v in state_dict.items():
        name = k[7:] # remove module.
        # name = k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)

    print('Inference...')
    f_txt = open(os.path.join(output_path, 'scores.txt'), 'w')
    total_psnr = []
    total_ssim = []
    PAD = cfg.TEST.pad
    t1 = time.time()
    for k in img_list:
        img1 = np.asarray(Image.open(os.path.join(img_path, str(k-1).zfill(4)+'.png')))
        img2 = np.asarray(Image.open(os.path.join(img_path, str(k+1).zfill(4)+'.png')))
        img1 = img1[np.newaxis, :, :]
        img1 = np.repeat(img1, 3, 0)
        img2 = img2[np.newaxis, :, :]
        img2 = np.repeat(img2, 3, 0)
        inputs = np.concatenate([img1, img2], axis=0)
        inputs = inputs[np.newaxis, :, :, :]
        inputs = inputs.astype(np.float32) / 255.0
        inputs = torch.from_numpy(inputs)

        inputs = inputs.to(device)
        inputs = F.pad(inputs, (PAD, PAD, PAD, PAD))
        with torch.no_grad():
            pred = model(inputs)
        pred = F.pad(pred, (-PAD, -PAD, -PAD, -PAD))
        pred = pred.data.cpu().numpy()
        pred = np.squeeze(pred)

        pred = (pred * 255).astype(np.uint8)
        Image.fromarray(pred).save(os.path.join(output_path, str(k).zfill(4)+'.png'))
        if args.mode == 'valid':
            gt_img = np.asarray(Image.open(os.path.join(img_path, str(k).zfill(4)+'.png')))
            _, psnr = compute_psnr(pred, gt_img)
            ssim = compute_ssim(pred, gt_img)
            total_psnr.append(psnr)
            total_ssim.append(ssim)
            print('image=%d, PSNR=%.4f, SSIM=%.4f' % (k, psnr, ssim))
            f_txt.write('image=%d, PSNR=%.4f, SSIM=%.4f' % (k, psnr, ssim))
            f_txt.write('\n')
    if args.mode == 'valid':
        mean_psnr = sum(total_psnr) / len(total_psnr)
        mean_ssim = sum(total_ssim) / len(total_ssim)
        print('mean_PSNR=%.4f, mean_SSIM=%.4f' % (mean_psnr, mean_ssim))
        f_txt.write('mean_PSNR=%.4f, mean_SSIM=%.4f' % (mean_psnr, mean_ssim))
    f_txt.close()
    print('COST TIME: ', (time.time() - t1))
