import os
import cv2
import numpy as np
from PIL import Image

class Provider_valid(object):
    def __init__(self, cfg, test=False):
        super(Provider_valid, self).__init__()
        if test == False:
            self.folder_name = cfg.DATA.folder_name
            self.valid_txt = cfg.DATA.valid_txt
        else:
            self.folder_name = cfg.TEST.folder_name
            self.valid_txt = cfg.TEST.valid_txt
        # read raw data
        f = open(os.path.join(self.folder_name, self.valid_txt), 'r')
        self.valid_list = [x[:-1] for x in f.readlines()]
    
    def read_img(self, img_id):
        img_name = img_id.split(' ')
        img1 = np.asarray(Image.open(os.path.join(self.folder_name, img_name[0])))
        img2 = np.asarray(Image.open(os.path.join(self.folder_name, img_name[1])))
        img3 = np.asarray(Image.open(os.path.join(self.folder_name, img_name[2])))
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0
        img3 = img3.astype(np.float32) / 255.0
        img1 = img1[np.newaxis, :, :]
        img2 = img2[np.newaxis, :, :]
        img3 = img3[np.newaxis, :, :]
        img1 = np.repeat(img1, 3, 0)
        img3 = np.repeat(img3, 3, 0)
        img_in = np.concatenate([img1, img3], axis=0)
        img_gt = img2
        return img_in, img_gt

    def __getitem__(self, index):
        img_id = self.valid_list[index]
        img_in, img_gt = self.read_img(img_id)
        return [img_in, img_gt]
    
    def __len__(self):
        return len(self.valid_list)