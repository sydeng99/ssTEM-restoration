import os
import cv2
import h5py
import numpy as np
from PIL import Image

class Provider_valid(object):
    def __init__(self, cfg, test=False):
        super(Provider_valid, self).__init__()
        if test == False:
            self.folder_name = cfg.DATA.folder_name
            self.valid_txt = cfg.DATA.valid_txt
            self.interp_valid_txt = cfg.DATA.interp_valid_txt
        else:
            self.folder_name = cfg.TEST.folder_name
            self.valid_txt = cfg.TEST.valid_txt
            self.interp_valid_txt = cfg.TEST.interp_valid_txt
        # read raw data
        f = open(os.path.join(self.folder_name, self.valid_txt), 'r')
        self.valid_list = [x[:-1] for x in f.readlines()]
        f.close()
        f = open(os.path.join(self.folder_name, self.interp_valid_txt), 'r')
        self.interp_valid_list = [x[:-1] for x in f.readlines()]
        f.close()
        self.gt_line = cfg.DATA.gt_line
    
    def read_img(self, index):
        img_id1 = self.valid_list[index]
        img_id2 = self.interp_valid_list[index]
        img_name = img_id1.split(' ')
        # img_name_split = img_name[1].split('/')
        # img_id3 = img_name_split[0]+ '/' +img_name_split[1].split('_')[0] + '_flow.hdf'
        # img1 = np.asarray(Image.open(os.path.join(self.folder_name, img_name[0])))
        img2 = np.asarray(Image.open(os.path.join(self.folder_name, img_name[1])))
        # img3 = np.asarray(Image.open(os.path.join(self.folder_name, img_name[2])))
        img_sff = np.asarray(Image.open(os.path.join(self.folder_name, img_name[3])))
        img_interp = np.asarray(Image.open(os.path.join(self.folder_name, img_id2)))
        # f = h5py.File(os.path.join(self.folder_name, img_id3), 'r')
        # flow2 = f['flow2'][:]
        # f.close()
        img_sff = img_sff[np.newaxis, :, :]
        img_sff = np.repeat(img_sff, 3, 0)
        img_interp = img_interp[np.newaxis, :, :]
        img_interp = np.repeat(img_interp, 3, 0)
        img_in = np.concatenate([img_sff, img_interp], axis=0)

        if self.gt_line:
            mask_line = np.ones_like(img_sff)
            mask_line[img_sff == 0] = 0
            img2 = np.multiply(img2, mask_line)
        
        img_in = img_in.astype(np.float32) / 255.0
        # img_gt = np.transpose(flow2, (2, 0, 1))
        img2 = img2[np.newaxis, :, :]
        img_gt = img2
        img_gt = img_gt.astype(np.float32) / 255.0
        return img_in, img_gt

    def __getitem__(self, index):
        img_in, img_gt = self.read_img(index)
        return [img_in, img_gt]
    
    def __len__(self):
        return len(self.valid_list)