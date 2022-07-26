from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import numpy as np
import random
from flow_synthesis import gen_line, gen_flow
from image_warp import image_warp
from flow_display import flow_to_image, sparse_flow, dense_flow
from skimage import io
import math

def SimuSFF(cleanImg_path,patch_size,save_folder, if_shave=True):
    filename=os.path.split(cleanImg_path)[-1]
    clean_img=io.imread(cleanImg_path)
    [h,w]=clean_img.shape
    if patch_size<h and patch_size<w:
        i=random.randint(0,h-patch_size)
        j=random.randint(0,w-patch_size)
        clean_patch=clean_img[i:i+patch_size,j:j+patch_size]
        sff_patch0, flow, mask=degradation(clean_patch,patch_size)
        flow_img=flow_to_image(flow)
        sff_patch=noise(sff_patch0,patch_size)
    else:
        sff_patch0, flow, mask=degradation(clean_img,h)
        flow_img=dense_flow(flow)
        sff_patch=noise(sff_patch0,h)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if if_shave:
        sff_patch=sff_patch[-1024:,-1024:]
        flow_img=flow_img[-1024:,-1024:]

    io.imsave(save_folder+filename.strip('.png')+'_SimuSFF.png',sff_patch)
    io.imsave(save_folder+filename.strip('.png')+'_SimuSFF_flow.png',flow_img)

def cal_distance(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def get_two_points(height,width,offset,crop_size):
    # two end points
    # 1 --> top line (0, x)
    # 2 --> right line (x, width)
    # 3 --> bottom line (height, x)
    # 4 --> left line (x, 0)
    k1 = random.randint(1, 4)
    k2 = random.randint(1, 4)
    while k1 == k2:
        k2 = random.randint(1, 4)

    if k1 == 1:
        x = random.randint(1, width - 1)
        while x < offset or x > crop_size - 50:
            x = random.randint(1, width - 1)
        p1 = [0, x]
    elif k1 == 2:
        x = random.randint(1, height - 1)
        while x < offset or x > crop_size - 50:
            x = random.randint(1, width - 1)
        p1 = [x, width]
    elif k1 == 3:
        x = random.randint(1, width - 1)
        while x < offset or x > crop_size - 50:
            x = random.randint(1, width - 1)
        p1 = [height, x]
    else:
        x = random.randint(1, height - 1)
        while x < offset or x > crop_size - 50:
            x = random.randint(1, width - 1)
        p1 = [x, 0]

    if k2 == 1:
        x = random.randint(1, width - 1)
        while x < offset or x > crop_size - 50:
            x = random.randint(1, width - 1)
        p2 = [0, x]
    elif k2 == 2:
        x = random.randint(1, height - 1)
        while x < offset or x > crop_size - 50:
            x = random.randint(1, width - 1)
        p2 = [x, width]
    elif k2 == 3:
        x = random.randint(1, width - 1)
        while x < offset or x > crop_size - 50:
            x = random.randint(1, width - 1)
        p2 = [height, x]
    else:
        x = random.randint(1, height - 1)
        while x < offset or x > crop_size - 50:
            x = random.randint(1, width - 1)
        p2 = [x, 0]

    return p1,p2

def degradation(img,crop_size,offset=50):
    flag = False
    while flag == False:
        height = crop_size
        width = crop_size
        line_width = random.randint(5, 20)
        # fold_width = random.randint(30, 80)
        fold_width = random.randint(10, 80)

        p1,p2=get_two_points(height,width,offset,crop_size)
        while cal_distance(p1,p2)<crop_size/2:
            p1, p2 = get_two_points(height, width, offset, crop_size)

        # dis_k = random.uniform(0.00000001, 0.0001)
        dis_k = random.uniform(0.00001, 0.1)
        k, b = gen_line(p1, p2)
        flow, mask = gen_flow(height, width, k, b, line_width, fold_width, dis_k)

        deformed = image_warp(img, flow, mode='bilinear')  # nearest or bilinear

        deformed = (deformed * mask).astype(np.uint8)
        # deformed = deformed[offset:-offset, offset:-offset]
        # mask = mask[self.offset:-self.offset, self.offset:-self.offset]

        pos = np.where(deformed == 0)
        count = len(pos[0])
        if count < 100:
            flag = False
        else:
            flag = True
    # deformed = deformed + (1 - mask) * random.randint(0, 60)
    # deformed = (deformed).astype(np.uint8)

    return deformed, flow, mask


def noise(img,det_size):
    mask = np.ones_like(img)
    mask[img == 0] = 0
    ran_reginal_contrast = random.uniform(0.4, 1.0)
    ran_w = random.randint(50, 200)
    ran_h = random.randint(50, 200)
    ran_pointx = random.randint(0, det_size - ran_h)
    ran_pointy = random.randint(0, det_size - ran_w)
    contrst_box = img[ran_pointx:ran_pointx + ran_h, ran_pointy:ran_pointy + ran_w]
    contrst_box = ran_reginal_contrast * (contrst_box - np.mean(img)) + np.mean(img)
    img[ran_pointx:ran_pointx + ran_h, ran_pointy:ran_pointy + ran_w] = contrst_box
    img = np.multiply(img, mask)
    return img

