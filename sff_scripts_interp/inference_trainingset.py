'''
Descripttion: 
version: 0.0
Author: Wei Huang
Date: 2022-03-14 16:41:05
'''
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

from data.provider_valid import Provider_valid
from model.model_interp import IFNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='ms_l1loss_decay')
    parser.add_argument('-id', '--model_id', type=str, default='interp')
    parser.add_argument('-t', '--test', action='store_false', default=True)
    parser.add_argument('-bs', '--batch_size', type=int, default=1)
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f))
    
    interp_train = 'interp_train_data'
    out_path = os.path.join(cfg.TEST.folder_name, interp_train)
    f_interp = open(os.path.join(cfg.TEST.folder_name, interp_train+'.txt'), 'w')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
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
    PAD = cfg.TEST.pad
    t1 = time.time()
    data_loder = Provider_valid(cfg, test=args.test)
    dataloader = torch.utils.data.DataLoader(data_loder, batch_size=args.batch_size)
    for k, data in enumerate(dataloader, 0):
        inputs, gt = data
        inputs = inputs.to(device)
        gt = gt.to(device)
        inputs = F.pad(inputs, (PAD, PAD, PAD, PAD))
        with torch.no_grad():
            pred = model(inputs)
        pred = F.pad(pred, (-PAD, -PAD, -PAD, -PAD))
        pred = pred.data.cpu().numpy()
        pred = np.squeeze(pred)

        pred = (pred * 255).astype(np.uint8)
        Image.fromarray(pred).save(os.path.join(out_path, str(k).zfill(4)+'_interp.png'))
        print(interp_train+'/'+str(k).zfill(4)+'_interp.png')
        f_interp.write(interp_train+'/'+str(k).zfill(4)+'_interp.png')
        f_interp.write('\n')
        f_interp.flush()
    f_interp.close()
    print('COST TIME: ', (time.time() - t1))
