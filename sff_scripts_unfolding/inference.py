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

from model.model_flownetC import FlowNetC
from model.model_flownetS import FlowNetS
from model.model_fusionnet import FusionNet
from utils.image_warp import image_warp
from utils.flow_display import dense_flow
from utils.image_warp_torch import SpatialTransformation
from loss.multiscaleloss import realEPE, EPE
from utils.psnr_ssim import compute_psnr, compute_ssim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='sff_flowfusionnet_L1_lr0001decay')
    parser.add_argument('-id', '--model_id', type=str, default='unfolding_fusionnet')
    parser.add_argument('-m', '--mode', type=str, default='valid')  # valid, test
    parser.add_argument('-ip', '--input_path', type=str, default='../data/test/test_cremic/')
    parser.add_argument('-t', '--txt_file', type=str, default='cremic_25sff')
    parser.add_argument('-op', '--output_path', type=str, default='../results/cremic')
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f))
    
    f_txt = open(os.path.join(args.input_path, args.txt_file+'.txt'), 'r')
    img_list = [int(x[:-1]) for x in f_txt.readlines()]
    f_txt.close()
    interp_path = os.path.join(args.output_path, args.txt_file+'_interp')
    sff_path = os.path.join(args.input_path, args.txt_file+'_degradation')

    output_path = os.path.join(args.output_path, args.txt_file+'_'+args.model_id)
    # unfolded_path = os.path.join(output_path, 'unfolded_img')
    unfolded_path = output_path
    flow_path = os.path.join(output_path, 'flow_img')
    if not os.path.exists(unfolded_path):
        os.makedirs(unfolded_path)
    if not os.path.exists(flow_path):
        os.makedirs(flow_path)


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Build FusionNet model
    if 'fusionnet' in args.model_id:
        print('Load FusionNet ...')
        model = FusionNet(input_nc=cfg.TRAIN.input_nc, output_nc=cfg.TRAIN.output_nc, ngf=cfg.TRAIN.ngf).to(device)
    elif 'flownetS' in args.model_id:
        print('Load FlowNetS ...')
        model = FlowNetS().to(device)
    elif 'flownetC' in args.model_id:
        print('Load FlowNetC ...')
        model = FlowNetC().to(device)
    else:
        raise AttributeError('No this mode!')

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
    model.eval()

    f_txt = open(os.path.join(output_path, 'scores.txt'), 'w')
    print('Inference...')
    PAD = cfg.TEST.pad
    warp = SpatialTransformation(use_gpu=torch.cuda.is_available())
    # sparse = cfg.TRAIN.sparse
    sparse = True
    flow_EPE = []
    total_psnr = []
    total_ssim = []
    pred_time = []
    warp_time = []
    t1 = time.time()
    for id_img in img_list:
        img_interp = np.asarray(Image.open(os.path.join(interp_path, str(id_img).zfill(4)+'.png')))
        img_sff = np.asarray(Image.open(os.path.join(sff_path, str(id_img).zfill(4)+'.png')))
        img_interp = img_interp[np.newaxis, :, :]
        img_interp = np.repeat(img_interp, 3, 0)
        img_sff = img_sff[np.newaxis, :, :]
        img_sff = np.repeat(img_sff, 3, 0)

        inputs = np.concatenate([img_sff, img_interp], axis=0)
        inputs = inputs[np.newaxis, :, :, :]
        inputs = inputs.astype(np.float32) / 255.0
        inputs = torch.from_numpy(inputs)

        inputs = inputs.to(device)
        inputs = F.pad(inputs, (PAD, PAD, PAD, PAD))
        start_pred = time.time()
        with torch.no_grad():
            pred_flow = model(inputs)
        end_pred = time.time()
        used_time = end_pred - start_pred
        pred_time.append(used_time)
        pred_flow = F.pad(pred_flow, (-PAD, -PAD, -PAD, -PAD))

        if args.mode == 'valid':
            flow2_path = os.path.join(args.input_path, args.txt_file+'_flow2', str(id_img).zfill(4) + '.hdf')
            f5 = h5py.File(flow2_path, 'r')
            gt_flow = f5['flow2'][:]
            f5.close()
            gt_flow = np.transpose(gt_flow, (2,0,1))
            gt_flow = gt_flow[np.newaxis, :, :, :]
            gt_flow = torch.from_numpy(gt_flow)
            gt_flow = gt_flow.to(device)
            b, _, h, w = gt_flow.size()
            if 'fusionnet' not in args.model_id:
                pred_flow = F.interpolate(pred_flow, (h,w), mode='bilinear', align_corners=False)
            epe_tmp = EPE(pred_flow, gt_flow, sparse, mean=True)
            flow_EPE.append(epe_tmp)
        
        input_sff = inputs[:, :3].clone()
        pred_flow = pred_flow.permute(0,2,3,1)
        t1 = time.time()
        warped_sff = warp(input_sff, pred_flow)
        t2 = time.time()
        tmp_warp = t2 - t1
        warp_time.append(tmp_warp)

        pred_flow = pred_flow.data.cpu().numpy()
        pred_flow = np.squeeze(pred_flow)
        flow_show = dense_flow(pred_flow)
        Image.fromarray(flow_show).save(os.path.join(flow_path, str(id_img).zfill(4)+'.png'))

        warped_img = ((np.squeeze(warped_sff.data.cpu().numpy())) * 255).astype(np.uint8)
        warped_img = np.transpose(warped_img, (1,2,0))
        Image.fromarray(warped_img).save(os.path.join(unfolded_path, str(id_img).zfill(4)+'.png'))
        if args.mode == 'valid':
            gt_img = np.asarray(Image.open(os.path.join(args.input_path, args.txt_file, str(id_img).zfill(4) + '.png')))
            warped_img = warped_img[:, :, 0]
            _, psnr = compute_psnr(warped_img, gt_img)
            ssim = compute_ssim(warped_img, gt_img)
            total_psnr.append(psnr)
            total_ssim.append(ssim)
            print('image=%d, EPE=%.4f, PSNR=%.4f, SSIM=%.4f' % (id_img, epe_tmp, psnr, ssim))
            f_txt.write('image=%d, EPE=%.4f, PSNR=%.4f, SSIM=%.4f' % (id_img, epe_tmp, psnr, ssim))
            f_txt.write('\n')
    if args.mode == 'valid':
        mean_EPE = sum(flow_EPE) / len(flow_EPE)
        mean_psnr = sum(total_psnr) / len(total_psnr)
        mean_ssim = sum(total_ssim) / len(total_ssim)
        print('mean_EPE=%.4f, mean_PSNR=%.4f, mean_SSIM=%.4f' % (mean_EPE, mean_psnr, mean_ssim))
        f_txt.write('mean_EPE=%.4f, mean_PSNR=%.4f, mean_SSIM=%.4f' % (mean_EPE, mean_psnr, mean_ssim))
        f_txt.write('\n')
    f_txt.close()
    ave_time = sum(pred_time) / len(pred_time)
    ave_warp = sum(warp_time) / len(warp_time)
    print('average inference time: %f' % ave_time)
    print('average warp time: %f' % ave_warp)
    print('COST TIME: ', (time.time() - t1))
