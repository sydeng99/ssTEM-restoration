from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import yaml
import time
import cv2
import logging
import argparse
import numpy as np
from PIL import Image
from attrdict import AttrDict
from tensorboardX import SummaryWriter
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data.data_provider import Provider
from data.provider_valid import Provider_valid
# from voxelmorph2d import VoxelMorph2d, vox_morph_loss
# from model.model_unet import UNet
from model.model_fusionnet import FusionNet
from loss.loss_ssim import MS_SSIM
from loss.loss_vgg import VGG19, vgg_loss
from utils.psnr_ssim import compute_psnr
from utils.flow_display import dense_flow
from utils.image_warp import image_warp


def init_project(cfg):
    def init_logging(path):
        logging.basicConfig(
                level    = logging.INFO,
                format   = '%(message)s',
                datefmt  = '%m-%d %H:%M',
                filename = path,
                filemode = 'w')

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    if cfg.TRAIN.is_cuda:
        if torch.cuda.is_available() is False:
            raise AttributeError('No GPU available')

    prefix = cfg.time
    if cfg.TRAIN.resume:
        model_name = cfg.TRAIN.model_name
    else:
        model_name = prefix + '_' + cfg.NAME
    cfg.cache_path = os.path.join(cfg.TRAIN.cache_path, model_name)
    cfg.save_path = os.path.join(cfg.TRAIN.save_path, model_name)
    # cfg.record_path = os.path.join(cfg.TRAIN.record_path, 'log')
    cfg.record_path = os.path.join(cfg.save_path, model_name)
    cfg.valid_path = os.path.join(cfg.save_path, 'valid')
    if cfg.TRAIN.resume is False:
        if not os.path.exists(cfg.cache_path):
            os.makedirs(cfg.cache_path)
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path)
        if not os.path.exists(cfg.record_path):
            os.makedirs(cfg.record_path)
        if not os.path.exists(cfg.valid_path):
            os.makedirs(cfg.valid_path)
    init_logging(os.path.join(cfg.record_path, prefix + '.log'))
    logging.info(cfg)
    writer = SummaryWriter(cfg.record_path)
    writer.add_text('cfg', str(cfg))
    return writer

def load_dataset(cfg):
    print('Caching datasets ... ', end='', flush=True)
    t1 = time.time()
    train_provider = Provider('train', cfg)
    if cfg.TRAIN.if_valid:
        valid_provider = Provider_valid(cfg)
    else:
        valid_provider = None
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider, valid_provider

def build_model(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')
    # model = UNet(in_channel=cfg.TRAIN.input_nc, out_channel=cfg.TRAIN.output_nc).to(device)
    model = FusionNet(input_nc=cfg.TRAIN.input_nc, output_nc=cfg.TRAIN.output_nc, ngf=cfg.TRAIN.ngf).to(device)

    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = nn.DataParallel(model)
        else:
            raise AttributeError('Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model

def resume_params(cfg, model, optimizer, resume):
    if resume:
        t1 = time.time()
        model_path = os.path.join(cfg.save_path, 'model-%06d.ckpt' % cfg.TRAIN.model_id)

        print('Resuming weights from %s ... ' % model_path, end='', flush=True)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_weights'])
            # optimizer.load_state_dict(checkpoint['optimizer_weights'])
        else:
            raise AttributeError('No checkpoint found at %s' % model_path)
        print('Done (time: %.2fs)' % (time.time() - t1))
        print('valid %d' % checkpoint['current_iter'])
        return model, optimizer, checkpoint['current_iter']
    else:
        return model, optimizer, 0

def calculate_lr(iters):
    if iters < cfg.TRAIN.warmup_iters:
        current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(float(iters) / cfg.TRAIN.warmup_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
    else:
        if iters < cfg.TRAIN.decay_iters:
            current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
        else:
            current_lr = cfg.TRAIN.end_lr
    return current_lr

def EPE(input_flow, target_flow, sparse=False, mean=True):
    EPE_map = torch.norm(target_flow-input_flow,2,1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/batch_size

def loop(cfg, train_provider, valid_provider, model, criterion, optimizer, iters, writer):
    model.train()
    PAD = cfg.TRAIN.pad
    TPAD = cfg.TEST.pad
    f_loss_txt = open(os.path.join(cfg.record_path, 'loss.txt'), 'a')
    f_valid_txt = open(os.path.join(cfg.record_path, 'valid.txt'), 'a')
    rcd_time = []
    sum_time = 0
    sum_loss = 0
    device = torch.device('cuda:0')

    if cfg.TRAIN.loss == 'L1':
        criterion = nn.L1Loss()
    elif cfg.TRAIN.loss == 'L2':
        criterion = F.mse_loss
    elif cfg.TRAIN.loss == 'ssim':
        criterion = MS_SSIM(max_val = 1)
    elif cfg.TRAIN.loss == 'perceptual':
        model_vgg = VGG19().to(device)
        cuda_count = torch.cuda.device_count()
        if cuda_count > 1:
            model_vgg = nn.DataParallel(model_vgg)
            print('VGG build on %d GPUs ... ' % cuda_count, flush=True)
        else:
            print('VGG build on a single GPU ... ', flush=True)
        vgg_weight = cfg.TRAIN.vgg_weight
        criterion_L1 = nn.L1Loss()
    else:
        raise AttributeError('No this loss function!')
    
    while iters <= cfg.TRAIN.total_iters:
        # train
        iters += 1
        t1 = time.time()
        input, target = train_provider.next()
        
        # decay learning rate
        if cfg.TRAIN.end_lr == cfg.TRAIN.base_lr:
            current_lr = cfg.TRAIN.base_lr
        else:
            current_lr = calculate_lr(iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
        optimizer.zero_grad()
        input = F.pad(input, (PAD, PAD, PAD, PAD))
        # pdb.set_trace()
        pred = model(input)
        pred = F.pad(pred, (-PAD, -PAD, -PAD, -PAD))

        if cfg.TRAIN.loss == 'perceptual':
            loss_L1 = criterion_L1(pred, target)
            pred_ = torch.cat([pred, pred, pred], dim=1)
            target_ = torch.cat([target, target, target], dim=1)
            out1 = model_vgg(pred_)
            out2 = model_vgg(target_)
            loss_vgg = vgg_loss(out1, out2, mode=1) * vgg_weight
            loss = loss_L1 + loss_vgg
        else:
            loss = criterion(pred, target)
            # loss = vox_morph_loss(pred_regd, target, n=cfg.TRAIN.n, lamda=cfg.TRAIN.lamda)
        loss.backward()
        if cfg.TRAIN.weight_decay is not None:
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data.add(-cfg.TRAIN.weight_decay * group['lr'], param.data)
        optimizer.step()
        
        sum_loss += loss.item()
        sum_time += time.time() - t1
        
        # log train
        if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
            rcd_time.append(sum_time)
            logging.info('step %d, loss = %.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                            % (iters, sum_loss / cfg.TRAIN.display_freq * 1, current_lr, sum_time,
                            (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(np.asarray(rcd_time)) / 60))
            if cfg.TRAIN.loss == 'perceptual':
                writer.add_scalar('loss/loss_L1', loss_L1, iters)
                writer.add_scalar('loss/loss_vgg', loss_vgg, iters)
            writer.add_scalar('loss', sum_loss / cfg.TRAIN.display_freq * 1, iters)
            f_loss_txt.write('step = ' + str(iters) + ', loss = ' + str(sum_loss / cfg.TRAIN.display_freq * 1))
            f_loss_txt.write('\n')
            f_loss_txt.flush()
            sys.stdout.flush()
            sum_time = 0
            sum_loss = 0
        
        # display
        if iters % cfg.TRAIN.valid_freq == 0 or iters == 1:
            input = F.pad(input, (-PAD, -PAD, -PAD, -PAD))
            input0 = ((np.squeeze(input[0].data.cpu().numpy())) * 255).astype(np.uint8)
            input1 = input0[0:3]
            input2 = input0[3:6]
            input1 = np.transpose(input1, (1,2,0))
            input2 = np.transpose(input2, (1,2,0))
            # target = (np.squeeze(target[0].data.cpu().numpy()) * 255).astype(np.uint8)
            # target = ((np.squeeze(target[0].data.cpu().numpy())) * 255).astype(np.uint8)
            target = np.squeeze(target[0].data.cpu().numpy())
            target = np.transpose(target, (1,2,0))
            pred = np.squeeze(pred[0].data.cpu().numpy())
            pred = np.transpose(pred, (1,2,0))
            pred_show = dense_flow(pred)
            target_show = dense_flow(target)
            # pred_regd[pred_regd>1] = 1; pred_regd[pred_regd<0] = 0
            # pred_regd = (pred_regd * 255).astype(np.uint8)
            im_cat1 = np.concatenate([input1, input2], axis=1)
            im_cat2 = np.concatenate([pred_show, target_show], axis=1)
            im_cat = np.concatenate([im_cat1, im_cat2], axis=0)
            Image.fromarray(im_cat).save(os.path.join(cfg.cache_path, '%06d.png' % iters))
        
        # valid
        if iters % cfg.TRAIN.save_freq == 0 or iters == 1:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            model.eval()
            running_loss = 0.0
            dataloader = torch.utils.data.DataLoader(valid_provider, batch_size=1)
            for k, data in enumerate(dataloader, 0):
                inputs, gt = data
                inputs = inputs.to(device)
                gt = gt.to(device)
                inputs = F.pad(inputs, (TPAD, TPAD, TPAD, TPAD))
                with torch.no_grad():
                    pred = model(inputs)
                pred = F.pad(pred, (-TPAD, -TPAD, -TPAD, -TPAD))
                loss_valid = EPE(pred, gt)
                pred = np.squeeze(pred.data.cpu().numpy())
                # print(pred_flow.shape)
                # pdb.set_trace()
                gt = np.squeeze(gt.data.cpu().numpy())
                # pred_regd[pred_regd>1] = 1; pred_regd[pred_regd<0] = 0
                # _, psnr = compute_psnr(pred_regd, gt)
                psnr = float(loss_valid)
                loss = psnr
                running_loss += loss
                if k == 0:
                    pred = np.transpose(pred, (1,2,0))
                    gt = np.transpose(gt, (1,2,0))
                    inputs = ((np.squeeze(inputs[0].data.cpu().numpy())) * 255).astype(np.uint8)
                    sff = inputs[0:3]
                    sff = np.transpose(sff, (1,2,0))
                    def_sff = image_warp(sff, pred)
                    im_cat1 = np.concatenate([sff, def_sff], axis=1)
                    flow_pred = dense_flow(pred)
                    flow_gt = dense_flow(gt)
                    im_cat2 = np.concatenate([flow_pred, flow_gt], axis=1)
                    im_cat = np.concatenate([im_cat1, im_cat2], axis=0)
                    Image.fromarray(im_cat).save(os.path.join(cfg.valid_path, str(iters).zfill(6)+'.png'))
            epoch_loss = running_loss / len(valid_provider)
            print('model-%d, valid-psnr=%.6f' % (iters, epoch_loss))
            writer.add_scalar('psnr', epoch_loss, iters)
            f_valid_txt.write('model-%d, valid-psnr=%.6f' % (iters, epoch_loss))
            f_valid_txt.write('\n')
            f_valid_txt.flush()

        # save
        if iters % cfg.TRAIN.save_freq == 0:
            states = {'current_iter': iters, 'valid_result': None,
                    'model_weights': model.state_dict()}
            torch.save(states, os.path.join(cfg.save_path, 'model-%06d.ckpt' % iters))
            print('***************save modol, iters = %d.***************' % (iters))
    f_loss_txt.close()
    f_valid_txt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='standard', help='path to config file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='path to config file')
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    print('mode: ' + args.mode)

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f))

    timeArray = time.localtime()
    time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S', timeArray)
    print('time stamp:', time_stamp)

    cfg.path = cfg_file
    cfg.time = time_stamp

    if args.mode == 'train':
        writer = init_project(cfg)
        train_provider, valid_provider = load_dataset(cfg)
        model = build_model(cfg, writer)
        optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
        # optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.base_lr, momentum=0.99)
        # optimizer = optim.Adamax(model.parameters(), lr=cfg.TRAIN.base_l, eps=1e-8)
        model, optimizer, init_iters = resume_params(cfg, model, optimizer, cfg.TRAIN.resume)
        loop(cfg, train_provider, valid_provider, model, nn.L1Loss(), optimizer, init_iters, writer)
        writer.close()
    else:
        pass
    print('***Done***')