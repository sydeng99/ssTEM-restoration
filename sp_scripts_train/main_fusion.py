#!/usr/bin/python3
import os
import numpy as np
from PIL import Image
import re
from time import time
from collections import OrderedDict
from utils.option import parse
from attrdict import AttrDict
import yaml
import argparse

from dataset import ImageDataset
from networks import UNet, IFNet, FusionNet

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable


parser = argparse.ArgumentParser()

parser.add_argument('-opt', type=str, default='./config/train_fusion.yaml',
                    help='Path to option opt file.') 
args = parser.parse_args()
opt = parse(args.opt, is_train=True)
print(opt)


# =============================================
#               building models
# =============================================
print('Building model on ', end='', flush=True)
t1 = time()
device = torch.device('cuda:0')
if opt['network_correction']=='Unet':
    model_denoise=UNet(1,1)
else:
    raise NotImplementedError

model_fusion=FusionNet(1,1)
    

model_denoise=model_denoise.to(device)
model_fusion=model_fusion.to(device)
model_vfi = IFNet().to(device)

cuda_count = torch.cuda.device_count()
if cuda_count > 1  and (opt['if_multiGPU']==True):
    if opt['batch_size'] % cuda_count == 0:
        print('%d GPUs ... ' % cuda_count, end='', flush=True)
        model_vfi = nn.DataParallel(model_vfi)
        model_denoise=nn.DataParallel(model_denoise)
        model_fusion=nn.DataParallel(model_fusion)
    else:
        raise AttributeError(
            'Batch size (%d) cannot be equally divided by GPU number (%d)' % (opt['batch_size'], cuda_count))
else:
    print('a single GPU ... ', end='', flush=True)
print('Done (time: %.2fs)' % (time() - t1))

if opt['if_pretrained_vfi']:
    lr_vfi=opt['base_lr']*opt['vfi_lr_weight']
else:
    lr_vfi=opt['base_lr']

if opt['if_pretrained_denoise']:
    lr_denoise=opt['base_lr']*opt['denoise_lr_weight']
else:
    lr_denoise=opt['base_lr']

optimizer_vfi = optim.Adam(model_vfi.parameters(), lr=lr_vfi, eps=1e-8)
optimizer_denoise = optim.Adam(model_denoise.parameters(), lr=lr_denoise, eps=1e-8)
optimizer_fusion = optim.Adam(model_fusion.parameters(), lr=opt['base_lr'], eps=1e-8)


# =============================================
#            reload or initialize
# =============================================
if opt['if_pretrained']:
    t1 = time()
    last_iter = 0
    for files in os.listdir(opt['resume_path']):
        if 'model' in files:
            it = int(re.sub('\D', '', files))
            if it > last_iter:
                last_iter = it
    if opt['if_pretrained_vfi']:
        model_vfi_path = os.path.join(opt['resume_path'], 'model_vfi-%d.ckpt' % opt['vfi_last_iter'])
    if opt['if_pretrained_denoise']:
        model_denoise_path = os.path.join(opt['resume_path'], 'model_denoise-%d.ckpt' % opt['denoise_last_iter'])
    if opt['if_pretrained_fusion']:
        model_fusion_path = os.path.join(opt['resume_path'], 'model_fusion-%d.ckpt' % last_iter)

    print('Resuming weights from %s ... ' % model_vfi_path, end='', flush=True)
    if os.path.isfile(model_vfi_path):
        if opt['if_pretrained_vfi']:
            checkpoint_vfi = torch.load(model_vfi_path)

            if opt['if_multiGPU']==True:
                model_vfi.module.load_state_dict(checkpoint_vfi['model_weights'])
            else:
                model_vfi.load_state_dict(checkpoint_vfi['model_weights'])
            optimizer_vfi.load_state_dict(checkpoint_vfi['optimizer_weights'])

        if opt['if_pretrained_denoise']:
            checkpoint_denoise=torch.load(model_denoise_path)
            # pretrained_dict = {k:v for k, v in checkpoint_denoise['model_weights'].items() if k in model_denoise.state_dict()}
            # model_denoise.load_state_dict(pretrained_dict)
            if opt['if_multiGPU'] == True:
                model_denoise.module.load_state_dict(checkpoint_denoise['model_weights'])
            else:
                # multi GPUs training, single GPU testing
                if opt['mGPUtrain_sGPUresume']:
                    new_checkpoint_weights = OrderedDict()
                    for k in checkpoint_denoise['model_weights'].keys():
                        if 'module' in k:
                            new_checkpoint_weights[k.replace('module.', '')] = checkpoint_denoise['model_weights'][k]
                    model_denoise.load_state_dict(new_checkpoint_weights)
                else:
                    model_denoise.load_state_dict(checkpoint_denoise['model_weights'])

            optimizer_denoise.load_state_dict(checkpoint_denoise['optimizer_weights'])

        if opt['if_pretrained_fusion']:
            checkpoint_fusion=torch.load(model_fusion_path)
            model_fusion.load_state_dict(checkpoint_fusion['model_weights'])
            optimizer_fusion.load_state_dict(checkpoint_fusion['optimizer_weights'])

        print('Done (time: %.2fs)' % (time() - t1))
        

# ======================================
#         losses & LR schedulers
# ======================================
if opt['loss_type']=='L2':
    criterion = torch.nn.MSELoss()
elif opt['loss_type']=='L1':
    criterion = torch.nn.L1Loss()
else:
    raise NotImplementedError

lr_scheduler_vfi=optim.lr_scheduler.StepLR(optimizer_vfi,30,gamma=0.5)
lr_scheduler_denoise=optim.lr_scheduler.StepLR(optimizer_denoise,30,gamma=0.5)
lr_scheduler_fusion=optim.lr_scheduler.StepLR(optimizer_fusion,30,gamma=0.5)

# =====================================
#              dataset
# =====================================
with open(args.opt, 'r') as f:
    cfg = AttrDict(yaml.load(f))

dataset=ImageDataset(cfg)
dataloader=DataLoader(dataset,batch_size=opt['batch_size'])
Tensor = torch.cuda.FloatTensor if opt['cuda'] else torch.Tensor

# ===================================
#             training
# ===================================
if not os.path.exists(opt['save_path']):
    os.makedirs(opt['save_path'])
if not os.path.exists(opt['cache_path']):
    os.makedirs(opt['cache_path'])


PAD = 0
f_loss_txt = open(os.path.join(opt['cache_path'], 'loss.txt'), 'w')
f_loss_txt.close()

for epoch in range(opt['epoch'],opt['n_epochs']):
    lr_scheduler_vfi.step(epoch)
    lr_scheduler_denoise.step(epoch)
    lr_scheduler_fusion.step(epoch)
    epoch_loss=0
    for i, batch in enumerate(dataloader):
        img_1 = batch['img_1']
        img_2 = batch['img_2']
        img_2_degra = batch['img_2_degra']
        img_3 = batch['img_3']
        img_3_degra = batch['img_3_degra']
        img_4 = batch['img_4']

        if opt['mode']=='a':
            img_2_degra_mask = batch['img_2_degraB1_mask_gradall']
            img_3_degra_mask = batch['img_3_degraB1_mask_gradall']
            img_2_degra_mask_r = batch['img_2_degraB1_mask_gradall_r']
            img_3_degra_mask_r = batch['img_3_degraB1_mask_gradall_r']
        elif opt['mode']=='b':
            img_2_degra_mask = batch['img_2_degraB1_GenGradMask']
            img_3_degra_mask = batch['img_3_degraB1_GenGradMask']
            img_2_degra_mask_r = batch['img_2_degraB1_GenGradMask_r']
            img_3_degra_mask_r = batch['img_3_degraB1_GenGradMask_r']


        input_1=torch.ones_like(img_2_degra_mask)
        img_2_degra_mask_r=input_1-img_2_degra_mask
        img_3_degra_mask_r=input_1-img_3_degra_mask

        optimizer_vfi.zero_grad()
        optimizer_denoise.zero_grad()
        optimizer_fusion.zero_grad()

        target1=img_2
        target2=img_3


        inputs_vfi = torch.cat((img_1, img_1, img_1, \
                                     img_4, img_4, img_4), 1)
        inputs_vfi=inputs_vfi.cuda()
        vfi_pred1 = torch.unsqueeze(model_vfi(inputs_vfi)[:, 0], 1)  # keep the dim in channel
        vfi_pred2 = torch.unsqueeze(model_vfi(inputs_vfi)[:, 1], 1)

        denoise_pred_1 = model_denoise(img_2_degra)
        denoise_pred_2 = model_denoise(img_3_degra)

        fusion_input1_1 = torch.mul(vfi_pred1, img_2_degra_mask_r)
        fusion_input1_2 = torch.mul(denoise_pred_1, img_2_degra_mask)
        fusion_input2_1 = torch.mul(vfi_pred2, img_3_degra_mask_r)
        fusion_input2_2 = torch.mul(denoise_pred_2, img_3_degra_mask)

        pred1 = model_fusion(fusion_input1_1, fusion_input1_2)
        pred2 = model_fusion(fusion_input2_1, fusion_input2_2)


        pred1 = F.pad(pred1, (-PAD, -PAD, -PAD, -PAD))
        pred2 = F.pad(pred2, (-PAD, -PAD, -PAD, -PAD))
        target1 = F.pad(target1, (-PAD, -PAD, -PAD, -PAD))
        target2 = F.pad(target2, (-PAD, -PAD, -PAD, -PAD))
        vfi_pred1 = F.pad(vfi_pred1, (-PAD, -PAD, -PAD, -PAD))
        vfi_pred2 = F.pad(vfi_pred2, (-PAD, -PAD, -PAD, -PAD))
        denoise_pred1 = F.pad(denoise_pred_1, (-PAD, -PAD, -PAD, -PAD))
        denoise_pred2 = F.pad(denoise_pred_2, (-PAD, -PAD, -PAD, -PAD))

        loss_vfi1 = criterion(vfi_pred1, target1)
        loss_vfi2 = criterion(vfi_pred2, target2)
        loss_denoise1 = criterion(denoise_pred1, target1)
        loss_denoise2 = criterion(denoise_pred2, target2)
        loss_fusion1 = criterion(pred1, target1)
        loss_fusion2 = criterion(pred2, target2)

        if opt['if_fusion_loss_only']:
            loss = loss_fusion1+loss_fusion2
        else:
            loss1 = loss_vfi1 + loss_denoise1 + loss_fusion1
            loss2 = loss_vfi2 + loss_denoise2 + loss_fusion2
            loss = loss1 + loss2


        epoch_loss += loss.item()
        loss.backward()

        optimizer_vfi.step()
        optimizer_denoise.step()
        optimizer_fusion.step()



        niter = epoch * len(dataloader) + i

        print_txt='nepoch:%d' % epoch+ '  niter:%d' % niter+ '  loss = %.6f' % loss.item()+', loss_vfi1 = %.6f' % loss_vfi1.item() \
                  + ', loss_denoise1 = %.6f' % loss_denoise1.item()  +', loss_fusion1 = %.6f' % loss_fusion1.item()


        print(print_txt)

        f_loss_txt = open(os.path.join(opt['cache_path'], 'loss.txt'), 'a')
        f_loss_txt.write(print_txt)

        f_loss_txt.write('\n')

        f_loss_txt.close()

        with torch.no_grad():
            if niter % opt['valid_freq'] == 0:
                pred1 = F.pad(pred1, (PAD, PAD, PAD, PAD))
                pred2 = F.pad(pred2, (PAD, PAD, PAD, PAD))
                target1 = F.pad(target1, (PAD, PAD, PAD, PAD))
                target2 = F.pad(target2, (PAD, PAD, PAD, PAD))
                vfi_pred1=F.pad(vfi_pred1,(PAD, PAD, PAD, PAD))
                vfi_pred2 = F.pad(vfi_pred2, (PAD, PAD, PAD, PAD))
                denoise_pred1=F.pad(denoise_pred1,(PAD, PAD, PAD, PAD))
                denoise_pred2 = F.pad(denoise_pred2, (PAD, PAD, PAD, PAD))

                input0 = (np.squeeze(inputs_vfi[0, 0].data.cpu().numpy()) * 255).astype(np.uint8)
                input3 = (np.squeeze(inputs_vfi[0, 3].data.cpu().numpy()) * 255).astype(np.uint8)
                input1_degra = (np.squeeze(img_2_degra[0][0].data.cpu().numpy()) * 255).astype(np.uint8)
                input2_degra = (np.squeeze(img_3_degra[0][0].data.cpu().numpy()) * 255).astype(np.uint8)

                target1 = (np.squeeze(target1[0, 0].data.cpu().numpy()) * 255).astype(np.uint8)
                target2 = (np.squeeze(target2[0, 0].data.cpu().numpy()) * 255).astype(np.uint8)
                pred_vfi1 = np.squeeze(vfi_pred1[0, 0].data.cpu().numpy())
                pred_vfi1[pred_vfi1 > 1] = 1
                pred_vfi1[pred_vfi1 < 0] = 0
                pred_vfi1 = (pred_vfi1 * 255).astype(np.uint8)
                pred_vfi2 = np.squeeze(vfi_pred2[0, 0].data.cpu().numpy())
                pred_vfi2[pred_vfi2 > 1] = 1
                pred_vfi2[pred_vfi2 < 0] = 0
                pred_vfi2 = (pred_vfi2 * 255).astype(np.uint8)

                pred_denoise1 = np.squeeze(denoise_pred1[0, 0].data.cpu().numpy())
                pred_denoise1[pred_denoise1 > 1] = 1
                pred_denoise1[pred_denoise1 < 0] = 0
                pred_denoise1 = (pred_denoise1 * 255).astype(np.uint8)
                pred_denoise2 = np.squeeze(denoise_pred2[0, 0].data.cpu().numpy())
                pred_denoise2[pred_denoise2 > 1] = 1
                pred_denoise2[pred_denoise2 < 0] = 0
                pred_denoise2 = (pred_denoise2 * 255).astype(np.uint8)

                pred_final1 = np.squeeze(pred1[0, 0].data.cpu().numpy())
                pred_final1[pred_final1 > 1] = 1
                pred_final1[pred_final1 < 0] = 0
                pred_final1 = (pred_final1 * 255).astype(np.uint8)
                pred_final2 = np.squeeze(pred2[0, 0].data.cpu().numpy())
                pred_final2[pred_final2 > 1] = 1
                pred_final2[pred_final2 < 0] = 0
                pred_final2 = (pred_final2 * 255).astype(np.uint8)

                im_cat1 = np.concatenate([input0, input1_degra, input2_degra, input3], axis=1)
                im_cat2 = np.concatenate([pred_vfi1, pred_denoise1,pred_final1,target1], axis=1)
                im_cat3 = np.concatenate([pred_vfi2, pred_denoise2, pred_final2, target2], axis=1)

                im_cat = np.concatenate([im_cat1, im_cat2,im_cat3], axis=0)
                Image.fromarray(im_cat).save(os.path.join(opt['cache_path'], '%06d.png' % niter))

        if niter % opt['save_freq'] == 0:
            states_vfi = {'current_iter': niter, 'valid_result': None,
                      'model_weights': model_vfi.state_dict()}
            torch.save(states_vfi, os.path.join(opt['save_path'], 'model_vfi-%d.ckpt' % niter))

            states_denoise = {'current_iter': niter, 'valid_result': None,
                      'model_weights': model_denoise.state_dict()}
            torch.save(states_denoise, os.path.join(opt['save_path'], 'model_denoise-%d.ckpt' % niter))

            states_fusion = {'current_iter': niter, 'valid_result': None,
                      'model_weights': model_fusion.state_dict()}
            torch.save(states_fusion, os.path.join(opt['save_path'], 'model_fusion-%d.ckpt' % niter))