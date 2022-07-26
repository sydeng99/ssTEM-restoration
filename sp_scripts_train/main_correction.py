#!/usr/bin/python3
import os
import argparse
import numpy as np
from attrdict import AttrDict
import yaml
from PIL import Image
import re
from time import time

from dataset import ImageDataset
from networks import UNet
from utils.option import parse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, default='./options_file/train_correc.yaml',
                    help='Path to option opt file.')
args = parser.parse_args()
opt = parse(args.opt, is_train=True)
print(opt)


# =============================================
#               building models
# =============================================
print('Building model on ', end='', flush=True)
t1 = time()
device = torch.device('cuda')
if opt['network']=='UNet':
    model_denoise=UNet(1,1)
else:
    raise NotImplementedError

model_denoise=model_denoise.to(device)

if opt['if_use_all_GPUs']==True:
    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if opt['batch_size'] % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model_denoise=nn.DataParallel(model_denoise)
        else:
            raise AttributeError(
                'Batch size (%d) cannot be equally divided by GPU number (%d)' % (opt['batch_size'], cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
else:
    model_denoise = model_denoise.to('cuda')
print('Done (time: %.2fs)' % (time() - t1))

optimizer_denoise = optim.Adam(model_denoise.parameters(), lr=opt['base_lr'], eps=1e-8)

# =============================================
#            reload or initialize
# =============================================
if opt['if_pretrained']:
    t1 = time()
    last_iter = 0
    for files in os.listdir(opt['save_path']):
        if 'model' in files:
            it = int(re.sub('\D', '', files))
            if it > last_iter:
                last_iter = it
    model_denoise_path = os.path.join(opt['resume_path'], 'model_denoise-%d.ckpt' % last_iter)

    print('Resuming weights from %s ... ' % model_denoise_path, end='', flush=True)
    if os.path.isfile(model_denoise_path):
        checkpoint_denoise=torch.load(model_denoise_path)
        model_denoise.load_state_dict(checkpoint_denoise['model_weights'])
        optimizer_denoise.load_state_dict(checkpoint_denoise['optimizer_weights'])

        print('Done (time: %.2fs)' % (time() - t1))
        print('valid %d, loss = %.4f' % (checkpoint_denoise['current_iter'], checkpoint_denoise['valid_result']))

# ======================================
#         losses & LR schedulers
# ======================================
if opt['loss_type_restore']=='L2':
    criterion_restore = torch.nn.MSELoss()
elif opt['loss_type_restore']=='L1':
    criterion_restore = torch.nn.L1Loss()
elif opt['loss_type_restore']=='CE':
    criterion_restore= torch.nn.CrossEntropyLoss()
elif opt['loss_type_restore']=='BCE':
    criterion_restore=torch.nn.BCELoss()
else:
    raise NotImplementedError

lr_scheduler_denoise=optim.lr_scheduler.StepLR(optimizer_denoise,30,gamma=0.5)

# =====================================
#              dataset
# =====================================
with open(args.opt, 'r') as f:
    cfg = AttrDict(yaml.load(f))

dataset=ImageDataset(cfg)
dataloader=DataLoader(dataset,batch_size=opt['batch_size'])
Tensor = torch.cuda.FloatTensor if opt['cuda'] else torch.Tensor
input_1=Tensor(opt['batch_size'], opt['input_nc'], opt['patch_size'], opt['patch_size'])


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
    lr_scheduler_denoise.step(epoch)
    epoch_loss=0
    for i, batch in enumerate(dataloader):

        f_loss_txt = open(os.path.join(opt['cache_path'], 'loss.txt'), 'a')

        img_1=batch['img_1']
        img_2=batch['img_2']
        img_2_degra=batch['img_2_degra']
        img_3=batch['img_3']
        img_3_degra=batch['img_3_degra']
        img_4=batch['img_4']

        optimizer_denoise.zero_grad()

        if opt['network']=='UNet':
            target1 = img_2
            target2 = img_3

            denoise_pred_1_restore = model_denoise(img_2_degra)
            denoise_pred_2_restore = model_denoise(img_3_degra)

            img_2_degra= F.pad(img_2_degra, (-PAD, -PAD, -PAD, -PAD))
            img_3_degra= F.pad(img_3_degra, (-PAD, -PAD, -PAD, -PAD))
            target1 = F.pad(target1, (-PAD, -PAD, -PAD, -PAD))
            target2 = F.pad(target2, (-PAD, -PAD, -PAD, -PAD))
            denoise_pred1 = F.pad(denoise_pred_1_restore, (-PAD, -PAD, -PAD, -PAD))
            denoise_pred2 = F.pad(denoise_pred_2_restore, (-PAD, -PAD, -PAD, -PAD))

            loss_denoise1_restore = criterion_restore(denoise_pred1, target1)
            loss_denoise2_restore = criterion_restore(denoise_pred2, target2)

            loss_restore = loss_denoise1_restore + loss_denoise2_restore
            loss =  loss_restore

            epoch_loss += loss.item()
            loss.backward()

            optimizer_denoise.step()
            lr_scheduler_denoise.step()
            niter = epoch * len(dataloader) + i

            print('nepoch:%d' % epoch, 'niter:%d' % niter, 'loss = %.6f' % loss.item(),
                    ', loss_restore = %.6f' % loss_restore.item())
            f_loss_txt.write('nepoch:%d' % epoch + 'niter:%d' % niter + 'loss =%.6f' % loss.item() +
                                ', loss_restore = %.6f' % loss_restore.item() )
            f_loss_txt.write('\n')


        with torch.no_grad():
            if niter % opt['valid_freq'] == 0:
                if opt['network']=='UNet':
                    target1 = F.pad(target1, (PAD, PAD, PAD, PAD))
                    target2 = F.pad(target2, (PAD, PAD, PAD, PAD))
                    denoise_pred1 = F.pad(denoise_pred1, (PAD, PAD, PAD, PAD))
                    denoise_pred2 = F.pad(denoise_pred2, (PAD, PAD, PAD, PAD))

                    input1_degra = (np.squeeze(img_2_degra[0][0].data.cpu().numpy()) * 255).astype(np.uint8)
                    input2_degra = (np.squeeze(img_3_degra[0][0].data.cpu().numpy()) * 255).astype(np.uint8)

                    target1 = (np.squeeze(target1[0, 0].data.cpu().numpy()) * 255).astype(np.uint8)
                    target2 = (np.squeeze(target2[0, 0].data.cpu().numpy()) * 255).astype(np.uint8)

                    pred_denoise1 = np.squeeze(denoise_pred1[0, 0].data.cpu().numpy())
                    pred_denoise1[pred_denoise1 > 1] = 1
                    pred_denoise1[pred_denoise1 < 0] = 0
                    pred_denoise1 = (pred_denoise1 * 255).astype(np.uint8)
                    pred_denoise2 = np.squeeze(denoise_pred2[0, 0].data.cpu().numpy())
                    pred_denoise2[pred_denoise2 > 1] = 1
                    pred_denoise2[pred_denoise2 < 0] = 0
                    pred_denoise2 = (pred_denoise2 * 255).astype(np.uint8)

                    im_cat1 = np.concatenate([input1_degra, input2_degra], axis=1)
                    im_cat2 = np.concatenate([pred_denoise1, pred_denoise2], axis=1)
                    im_cat3 = np.concatenate([target1, target2], axis=1)

                    im_cat = np.concatenate([im_cat1, im_cat2, im_cat3], axis=0)
                    Image.fromarray(im_cat).save(os.path.join(opt['cache_path'], '%06d.png' % niter))

                else:
                    raise NotImplementedError


        if niter % opt['save_freq'] == 0:
            if opt['only_save_weights']==True:
                states_denoise = {'current_iter': niter,
                                  'model_weights': model_denoise.state_dict()}
                torch.save(states_denoise, os.path.join(opt['save_path'], 'model_denoise-%d.ckpt' % niter))
            else:
                states_denoise = {'current_iter': niter, 'valid_result': None,
                          'model_weights': model_denoise.state_dict(), 'optimizer_weights': optimizer_denoise.state_dict()}
                torch.save(states_denoise, os.path.join(opt['save_path'], 'model_denoise-%d.ckpt' % niter))


        f_loss_txt.close()