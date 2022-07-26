#!/usr/bin/python3
import os
import numpy as np
from PIL import Image
import re
from time import time
from attrdict import AttrDict
import yaml
import argparse

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import torch.optim as optim
import torch.nn as nn

from dataset import ImageDataset
from networks import IFNet
from utils.option import parse



parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, default='./config/train_interp.yaml',
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
model_vfi = IFNet().to(device)

cuda_count = torch.cuda.device_count()
if cuda_count > 1  and (opt['if_multiGPU']==True):
    if opt['batch_size'] % cuda_count == 0:
        print('%d GPUs ... ' % cuda_count, end='', flush=True)
        model_vfi = nn.DataParallel(model_vfi)
    else:
        raise AttributeError(
            'Batch size (%d) cannot be equally divided by GPU number (%d)' % (opt['batch_size'], cuda_count))
else:
    print('a single GPU ... ', end='', flush=True)
print('Done (time: %.2fs)' % (time() - t1))

lr_vfi=opt['base_lr']

optimizer_vfi = optim.Adam(model_vfi.parameters(), lr=lr_vfi, eps=1e-8)


# =============================================
#            reload or initialize
# =============================================
if opt['if_pretrained_vfi']:
    t1 = time()
    last_iter = 0
    for files in os.listdir(opt['resume_path']):
        if 'model' in files:
            it = int(re.sub('\D', '', files))
            if it > last_iter:
                last_iter = it
    if opt['if_pretrained_vfi']:
        model_vfi_path = os.path.join(opt['resume_path'], 'model_vfi-%d.ckpt' % opt['vfi_last_iter'])
        print('Resuming weights from %s ... ' % model_vfi_path, end='', flush=True)

        if os.path.isfile(model_vfi_path):
            if opt['if_pretrained_vfi']:
                checkpoint_vfi = torch.load(model_vfi_path)
                if opt['if_multiGPU']==True:
                    model_vfi.module.load_state_dict(checkpoint_vfi['model_weights'])
                else:
                    model_vfi.load_state_dict(checkpoint_vfi['model_weights'])
                optimizer_vfi.load_state_dict(checkpoint_vfi['optimizer_weights'])


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

# =====================================
#              dataset
# =====================================
with open(args.opt, 'r') as f:
    cfg = AttrDict(yaml.load(f))

dataset=ImageDataset(cfg)
dataloader=DataLoader(dataset,batch_size=opt['batch_size'])

# ===================================
#             training
# ===================================
if not os.path.exists(opt['save_path']):
    os.makedirs(opt['save_path'])
if not os.path.exists(opt['cache_path']):
    os.makedirs(opt['cache_path'])


PAD = opt['PAD']
f_loss_txt = open(os.path.join(opt['cache_path'], 'loss.txt'), 'w')
f_loss_txt.close()
for epoch in range(opt['epoch'],opt['n_epochs']):
    lr_scheduler_vfi.step(epoch)
    epoch_loss=0
    for i, batch in enumerate(dataloader):
        img_1 = batch['img_1']
        img_2 = batch['img_2']
        img_3 = batch['img_3']
        img_4 = batch['img_4']

        optimizer_vfi.zero_grad()

        target1=img_2
        target2=img_3

        inputs_vfi = torch.cat((img_1, img_1, img_1, \
                                     img_4, img_4, img_4), 1)
        inputs_vfi=inputs_vfi.cuda()
        vfi_pred1 = torch.unsqueeze(model_vfi(inputs_vfi)[:, 0], 1)  # keep the dim in channel
        vfi_pred2 = torch.unsqueeze(model_vfi(inputs_vfi)[:, 1], 1)


        target1 = F.pad(target1, (-PAD, -PAD, -PAD, -PAD))
        target2 = F.pad(target2, (-PAD, -PAD, -PAD, -PAD))
        vfi_pred1 = F.pad(vfi_pred1, (-PAD, -PAD, -PAD, -PAD))
        vfi_pred2 = F.pad(vfi_pred2, (-PAD, -PAD, -PAD, -PAD))

        loss_vfi1 = criterion(vfi_pred1, target1)
        loss_vfi2 = criterion(vfi_pred2, target2)

        loss = loss_vfi1 + loss_vfi2


        epoch_loss += loss.item()
        loss.backward()

        optimizer_vfi.step()

        niter = epoch * len(dataloader) + i

        print_txt='nepoch:%d' % epoch+ '  niter:%d' % niter+ '  loss = %.6f' % loss.item()+', loss_vfi1 = %.6f' % loss_vfi1.item()

        print(print_txt)
        f_loss_txt = open(os.path.join(opt['cache_path'], 'loss.txt'), 'a')
        f_loss_txt.write(print_txt)
        f_loss_txt.write('\n')

        f_loss_txt.close()

        with torch.no_grad():
            if niter % opt['valid_freq'] == 0:
                target1 = F.pad(target1, (PAD, PAD, PAD, PAD))
                target2 = F.pad(target2, (PAD, PAD, PAD, PAD))
                vfi_pred1=F.pad(vfi_pred1,(PAD, PAD, PAD, PAD))
                vfi_pred2 = F.pad(vfi_pred2, (PAD, PAD, PAD, PAD))

                input0 = (np.squeeze(inputs_vfi[0, 0].data.cpu().numpy()) * 255).astype(np.uint8)
                input3 = (np.squeeze(inputs_vfi[0, 3].data.cpu().numpy()) * 255).astype(np.uint8)

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


                im_cat1 = np.concatenate([input0, input3], axis=1)
                im_cat2 = np.concatenate([pred_vfi1,target1], axis=1)
                im_cat3 = np.concatenate([pred_vfi2, target2], axis=1)

                im_cat = np.concatenate([im_cat1, im_cat2,im_cat3], axis=0)
                Image.fromarray(im_cat).save(os.path.join(opt['cache_path'], '%06d.png' % niter))

        if niter % opt['save_freq'] == 0:
            states_vfi = {'current_iter': niter, 'valid_result': None,
                      'model_weights': model_vfi.state_dict(), 'optimizer_weights': optimizer_vfi.state_dict()}
            torch.save(states_vfi, os.path.join(opt['save_path'], 'model_vfi-%d.ckpt' % niter))

