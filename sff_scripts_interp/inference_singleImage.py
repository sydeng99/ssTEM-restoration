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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='ms_l1loss_decay')
    parser.add_argument('-id', '--model_id', type=str, default='interp')
    parser.add_argument('-i1', '--img1', type=str, default=None)
    parser.add_argument('-i2', '--img2', type=str, default=None)
    parser.add_argument('-o', '--output', type=str, default=None)
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f))
    
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
    img1 = np.asarray(Image.open(args.img1))
    img1 = img1[np.newaxis, :, :]
    img1 = np.repeat(img1, 3, 0)

    img2 = np.asarray(Image.open(args.img2))
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

    Image.fromarray(pred).save(args.output)
    print('COST TIME: ', (time.time() - t1))
