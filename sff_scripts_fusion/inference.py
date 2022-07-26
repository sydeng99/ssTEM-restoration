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
from model.model_unet import UNet
from utils.image_warp import image_warp
from utils.flow_display import dense_flow
from utils.image_warp_torch import SpatialTransformation
from utils.psnr_ssim import compute_psnr, compute_ssim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='sff_fusion_L1_lr0001decay')
    parser.add_argument('-id', '--model_id', type=str, default='fusion')
    parser.add_argument('-fm', '--flow_model', type=str, default='unfolding_fusionnet')
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
    fusion_path = output_path
    flow_path = os.path.join(output_path, 'flow_img')
    stitching_path = os.path.join(output_path, 'stitching_img')
    if not os.path.exists(fusion_path):
        os.makedirs(fusion_path)
    if not os.path.exists(flow_path):
        os.makedirs(flow_path)
    if not os.path.exists(stitching_path):
        os.makedirs(stitching_path)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Build FlowNet model
    if 'fusionnet' in args.flow_model:
        print('Load FusionNet ...')
        model_flow = FusionNet(input_nc=6, output_nc=2, ngf=32).to(device)
    elif 'flownetS' in args.flow_model:
        print('Load FlowNetS ...')
        model_flow = FlowNetS().to(device)
    elif 'flownetC' in args.flow_model:
        print('Load FlowNetC ...')
        model_flow = FlowNetC().to(device)
    else:
        raise AttributeError('No this mode!')

    ckpt_path = os.path.join('../trained_models', args.flow_model, args.flow_model+'.ckpt')
    checkpoint = torch.load(ckpt_path)

    new_state_dict = OrderedDict()
    state_dict = checkpoint['model_weights']
    for k, v in state_dict.items():
        name = k[7:] # remove module.
        # name = k
        new_state_dict[name] = v
    model_flow.load_state_dict(new_state_dict)
    model_flow.eval()

    # Build Fusion (UNet) model
    print('Load Fusion module...')
    model_fusion = UNet(in_channel=cfg.TRAIN.input_nc, out_channel=cfg.TRAIN.output_nc).to(device)
    # model_fusion = nn.DataParallel(model_fusion)

    ckpt_path = os.path.join('../trained_models', args.model_id, args.model_id+'.ckpt')
    checkpoint = torch.load(ckpt_path)

    new_state_dict = OrderedDict()
    state_dict = checkpoint['model_weights']
    for k, v in state_dict.items():
        name = k[7:] # remove module.
        # name = k
        new_state_dict[name] = v
    # new_state_dict= state_dict

    delkeys = []
    for k in new_state_dict:
        if 'batches_tracked' in k:
            delkeys.append(k)
    for delk in delkeys:
        new_state_dict.__delitem__(delk)

    model_fusion.load_state_dict(new_state_dict)
    model_fusion = model_fusion.to(device)
    model_fusion.eval()

    f_txt = open(os.path.join(output_path, 'scores.txt'), 'w')
    print('Inference...')
    warp = SpatialTransformation(use_gpu=torch.cuda.is_available())
    PAD = cfg.TEST.pad
    total_psnr = []
    total_ssim = []
    pred_time = []
    flow_time = []
    warp_time = []
    fusion_time = []
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
            pred_flow = model_flow(inputs)
            t1 = time.time()
            # input_sff = inputs[:, :3].detach()
            input_sff = inputs[:, :3].clone()
            if 'fusionnet' not in args.flow_model:
                b, _, h, w = inputs.size()
                pred_flow = F.interpolate(pred_flow, (h,w), mode='bilinear', align_corners=False)
            pred_flow = pred_flow.permute(0,2,3,1)
            warped_sff = warp(input_sff, pred_flow)
            t2 = time.time()
            inputs[:, :3] = warped_sff
            pred = model_fusion(inputs)
        end_pred = time.time()
        used_time = end_pred - start_pred
        pred_time.append(used_time)
        flow_time.append(t1-start_pred)
        warp_time.append(t2-t1)
        fusion_time.append(end_pred-t2)
        pred = F.pad(pred, (-PAD, -PAD, -PAD, -PAD))
        pred_flow = pred_flow.data.cpu().numpy()
        pred_flow = np.squeeze(pred_flow)
        pred = (np.squeeze(pred.data.cpu().numpy()) * 255).astype(np.uint8)
        warped_sff = (np.squeeze(warped_sff.data.cpu().numpy()) * 255).astype(np.uint8)
        warped_sff = np.transpose(warped_sff, (1,2,0))
        warped_sff = np.asarray(Image.fromarray(warped_sff).convert('L'))
        mask = np.ones_like(warped_sff, dtype=np.float32)
        mask[warped_sff < 2] = 0
        img_interp = img_interp[0]
        stitch = img_interp * (1 - mask) + warped_sff * mask
        stitch = stitch.astype(np.uint8)
        flow_show = dense_flow(pred_flow)
        
        if args.mode == 'valid':
            img2_path = os.path.join(args.input_path, args.txt_file, str(id_img).zfill(4) + '.png')
            img2 = np.asarray(Image.open(img2_path))
            _, psnr = compute_psnr(pred, img2)
            ssim = compute_ssim(pred, img2)
            total_psnr.append(psnr)
            total_ssim.append(ssim)
            print('image=%d, PSNR=%.4f, SSIM=%.4f' % (id_img, psnr, ssim))
            f_txt.write('image=%d, PSNR=%.4f, SSIM=%.4f' % (id_img, psnr, ssim))
            f_txt.write('\n')
        Image.fromarray(flow_show).save(os.path.join(flow_path, str(id_img).zfill(4)+'.png'))
        Image.fromarray(pred).save(os.path.join(fusion_path, str(id_img).zfill(4)+'.png'))
        Image.fromarray(stitch).save(os.path.join(stitching_path, str(id_img).zfill(4)+'.png'))
    if args.mode == 'valid':
        mean_psnr = sum(total_psnr) / len(total_psnr)
        mean_ssim = sum(total_ssim) / len(total_ssim)
        print('mean_PSNR=%.4f, mean_SSIM=%.4f' % (mean_psnr, mean_ssim))
        f_txt.write('mean_PSNR=%.4f, mean_SSIM=%.4f' % (mean_psnr, mean_ssim))
        f_txt.write('\n')
    f_txt.close()
    ave_time = sum(pred_time) / len(pred_time)
    ave_flow = sum(flow_time) / len(flow_time)
    ave_warp = sum(warp_time) / len(warp_time)
    ave_fusion = sum(fusion_time) / len(fusion_time)
    print('average inference time: %f' % ave_time)
    print('average flow time: %f' % ave_flow)
    print('average warp time: %f' % ave_warp)
    print('average fusin time: %f' % ave_fusion)
    print('COST TIME: ', (time.time() - t1))
