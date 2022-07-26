import os
import sys
import cv2
import time
import random
import numpy as np
from PIL import Image, ImageEnhance
from scipy.ndimage import binary_erosion
from skimage import io
import time
import extractM


def judge_bigMask_dis_legal(map,x1,y1,x2,y2,x3,y3,x4,y4,threshold,num):
    ret, binary = cv2.threshold(map, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(map,contours,-1,200,10)
    cv2.circle(map,(x1,y1),5,(255,255,255),-1)
    flag0=0
    for i in range(len(contours)):
        flag1=cv2.pointPolygonTest(contours[i], (x1,y1), True)
        flag2 = cv2.pointPolygonTest(contours[i], (y2, x2), True)
        flag3 = cv2.pointPolygonTest(contours[i], (y3, x3), True)
        flag4 = cv2.pointPolygonTest(contours[i], (y4, x4), True)
        if flag1>threshold and flag2>threshold and flag3>threshold and flag4>threshold:
            flag0+=1
    if flag0==0 :
        OK=True
    else:
        OK=False
    return OK


def BCA_acjust(oriImg,alpha,beta):
    newimg=oriImg*alpha+beta
    return newimg


def multiply_mask(out_img,mask,mask10,mask01,h,w,h_mask,w_mask,mask_map,mask_map_contour,mask_map_contour_grad,mask_map_contour_big,if_bigMask,num, area):

    flag=False
    if out_img.ndim==3:
        out_img=out_img[:,:,0]
    if mask.ndim==3:
        mask=mask[:,:,0]
    if mask10.ndim==3:
        mask10=mask10[:,:,0]
    if mask01.ndim==3:
        mask01=mask01[:,:,0]

    while flag==False:
        ran_h = random.randint(0, h-1)
        ran_w = random.randint(0, w-1)
        end_h = ran_h + h_mask
        end_w = ran_w + w_mask
        if end_h<h and end_w < w:
            x1,y1=ran_h,ran_w
            x2,y2=ran_h,end_w
            x3,y3=end_h,ran_w
            x4,y4=end_h,end_w
        elif end_h < h and end_w >= w:
            x1,y1=ran_h,ran_w
            x2,y2=ran_h,w-1
            x3,y3=end_h,ran_w
            x4,y4=end_h,w-1
        elif end_h >= h and end_w < w:
            x1,y1=ran_h,ran_w
            x2,y2=ran_h,end_w
            x3,y3=h-1,ran_w
            x4,y4=h-1,end_w
        else:
            x1,y1=ran_h,ran_w
            x2,y2=ran_h,w-1
            x3,y3=h-1,ran_w
            x4,y4=h-1,w-1


        newmap=mask_map.copy()
        channel=1
        if end_h <= h and end_w <= w:
            newmap[ran_h:end_h, ran_w:end_w] = 255.
            only_maskmap_sum=channel*h_mask*w_mask*255.
        elif end_h <= h and end_w > w:
            newmap[ran_h:end_h, ran_w:w] = 255.
            only_maskmap_sum=channel*h_mask*(w_mask - (end_w - w))*255.
        elif end_h > h and end_w <= w:
            newmap[ran_h:h, ran_w:end_w] = 255.
            only_maskmap_sum=channel*(h_mask - (end_h - h))*w_mask*255.
        else:
            newmap[ran_h:h, ran_w:w] = 255.
            only_maskmap_sum=channel*(h_mask - (end_h - h))*(w_mask - (end_w - w))*255.

        flag0 = (np.sum(newmap) == np.sum(mask_map) + only_maskmap_sum)

        if not if_bigMask:
            flag=flag0
        else:
            flag=flag0 and judge_bigMask_dis_legal(mask_map,x1,y1,x2,y2,x3,y3,x4,y4,-250,num)


    out_img1=out_img.copy()
    out_img2=out_img.copy()
    if end_h <= h and end_w <= w:
        out_img1[ran_h:end_h, ran_w:end_w] = np.multiply(out_img1[ran_h:end_h, ran_w:end_w], mask10)
        out_img2[ran_h:end_h,ran_w:end_w]=np.multiply(out_img2[ran_h:end_h, ran_w:end_w], mask01)
        out_img2[ran_h:end_h,ran_w:end_w]=np.multiply(out_img2[ran_h:end_h,ran_w:end_w], (mask/255.).astype('float'))
        out_img[ran_h:end_h, ran_w:end_w] = np.add(out_img1[ran_h:end_h, ran_w:end_w], out_img2[ran_h:end_h,ran_w:end_w])

        mask_map[ran_h:end_h, ran_w:end_w] = 255.
        mask_map_contour[ran_h:end_h, ran_w:end_w]=mask01
        mask_map_contour_grad[ran_h:end_h, ran_w:end_w]=mask


        if area>60000:
            mask_map_contour_big[ran_h:end_h, ran_w:end_w] = mask01

    elif end_h <= h and end_w > w:
        out_img1[ran_h:end_h, ran_w:w] = np.multiply(out_img1[ran_h:end_h, ran_w:w], mask10[:, 0:w_mask - (end_w - w)])
        out_img2[ran_h:end_h, ran_w:w] = np.multiply(out_img2[ran_h:end_h, ran_w:w], mask01[:, 0:w_mask - (end_w - w)])
        out_img2[ran_h:end_h,ran_w:w] = np.multiply(out_img2[ran_h:end_h, ran_w:w], (mask[:, 0:w_mask - (end_w - w)]/255.).astype('float'))
        out_img[ran_h:end_h, ran_w:w] = np.add(out_img1[ran_h:end_h, ran_w:w],out_img2[ran_h:end_h, ran_w:w])

        mask_map[ran_h:end_h, ran_w:w] = 255.
        mask_map_contour[ran_h:end_h, ran_w:w]=mask01[:, 0:w_mask - (end_w - w)]
        mask_map_contour_grad[ran_h:end_h, ran_w:w]=mask[:, 0:w_mask - (end_w - w)]
        if area>60000:
            mask_map_contour_big[ran_h:end_h, ran_w:w]=mask01[:, 0:w_mask - (end_w - w)]

    elif end_h > h and end_w <= w:
        out_img1[ran_h:h, ran_w:end_w] = np.multiply(out_img1[ran_h:h, ran_w:end_w], mask10[0:h_mask - (end_h - h), :])
        out_img2[ran_h:h, ran_w:end_w] = np.multiply(out_img2[ran_h:h, ran_w:end_w], mask01[0:h_mask - (end_h - h), :])
        out_img2[ran_h:h, ran_w:end_w] = np.multiply(out_img2[ran_h:h, ran_w:end_w] , (mask[0:h_mask - (end_h - h), :]/255.).astype('float'))
        out_img[ran_h:h, ran_w:end_w] =np.add(out_img1[ran_h:h, ran_w:end_w],out_img2[ran_h:h, ran_w:end_w])
        mask_map[ran_h:h, ran_w:end_w]=255.
        mask_map_contour[ran_h:h, ran_w:end_w]=mask01[0:h_mask - (end_h - h), :]
        mask_map_contour_grad[ran_h:h, ran_w:end_w]=mask[0:h_mask - (end_h - h), :]
        if area>60000:
            mask_map_contour_big[ran_h:h, ran_w:end_w]=mask01[0:h_mask - (end_h - h), :]

    else:
        out_img1[ran_h:h, ran_w:w] = np.multiply(out_img1[ran_h:h, ran_w:w],
                                                mask10[0:h_mask - (end_h - h), 0:w_mask - (end_w - w)])
        out_img2[ran_h:h, ran_w:w] = np.multiply(out_img2[ran_h:h, ran_w:w],
                                           mask01[0:h_mask - (end_h - h), 0:w_mask - (end_w - w)])
        out_img2[ran_h:h, ran_w:w] = np.multiply(out_img2[ran_h:h, ran_w:w], (mask[0:h_mask - (end_h - h), 0:w_mask - (end_w - w)]/255.).astype('float'))
        out_img[ran_h:h, ran_w:w]=np.add(out_img1[ran_h:h, ran_w:w],out_img2[ran_h:h, ran_w:w])

        mask_map[ran_h:h,ran_w:w]=255.
        mask_map_contour[ran_h:h,ran_w:w]=mask01[0:h_mask - (end_h - h), 0:w_mask - (end_w - w)]
        mask_map_contour_grad[ran_h:h,ran_w:w]=mask[0:h_mask - (end_h - h), 0:w_mask - (end_w - w)]
        if area>60000:
            mask_map_contour_big[ran_h:h,ran_w:w]=mask01[0:h_mask - (end_h - h), 0:w_mask - (end_w - w)]

    return out_img,mask_map,mask_map_contour,mask_map_contour_grad, mask_map_contour_big


def SimuSP(img, maskbank_root, mask10_root, mask01_root, area_stat=[5000,15000,30000,80000], area_nums_stat=[10,8,5,3,1]):
    aaa=random.uniform(0.3,0.5)
    bbb=170-147*aaa
    out_img=BCA_acjust(img,aaa,bbb)

    if out_img.ndim==3:
        [h,w,c]=out_img.shape
    else:
        [h,w]=out_img.shape

    masks=os.listdir(maskbank_root)
    nums=0
    num=[0]*len(area_nums_stat)
    num_masks = 1000

    masks_area=[]
    masks_ids=[]

    maskmap=np.zeros_like(img)
    maskmap_contour=np.zeros_like(img)
    mask_map_contour_grad=np.zeros_like(img)
    maskmap_contour_big=np.zeros_like(img)
    while nums<=num_masks:
        mask_id=random.randint(0,len(masks)-1)
        mask_path=mask_root+masks[mask_id]
        if masks[mask_id]!='mask0.png' and os.path.isfile(mask_path):
            mask=io.imread(mask_path)[:,:]
            mask10p=masks[mask_id]
            if mask.ndim==3:
                gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            else:
                gray=mask
            ret, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            area = []
            for k in range(len(contours)):
                area.append(cv2.contourArea(contours[k]))
            area2 = sorted(area)
            contour_i=area.index(area2[-1])
            area=cv2.contourArea(contours[contour_i])

            if area<area_stat[0]:
                if num[0] <= area_nums_stat[0]:
                    masks_area.append(area)
                    masks_ids.append(mask_id)
                    num[0] += 1
            elif area>=area_stat[0] and area<area_stat[1]:
                if num[1] <= area_nums_stat[1]:
                    masks_area.append(area)
                    masks_ids.append(mask_id)
                    num[1] += 1
            elif area>=area_stat[1] and area<area_stat[2]:
                if num[2] <= area_nums_stat[2]:
                    masks_area.append(area)
                    masks_ids.append(mask_id)
                    num[2] += 1
            elif area>=area_stat[2] and area<area_stat[3]:
                if num[3] <= area_nums_stat[3]:
                    masks_area.append(area)
                    masks_ids.append(mask_id)
                    num[3] += 1
            elif area>=area_stat[3]:
                if num[4] <= area_nums_stat[4]:
                    masks_area.append(area)
                    masks_ids.append(mask_id)
                    num[4] += 1
            nums += 1

    masks_area_sorted, masks_ids_sorted = (list(t) for t in zip(*sorted(zip(masks_area, masks_ids))))
    masks_area_sorted.reverse()
    masks_ids_sorted.reverse()

    for hhh in range(len(masks_area_sorted)):
        area=masks_area_sorted[hhh]
        mask_id=masks_ids_sorted[hhh]
        mask_path = mask_root + masks[mask_id]
        mask = io.imread(mask_path)[:, :]
        mask10p = masks[mask_id]

        if area>80000:
            if_bigMask=True
        else:
            if_bigMask=False


        if mask.ndim==3:
            mask = mask[:, :, 0]
        mask10 = io.imread(mask10_root + mask10p)/ 255.
        mask01 = io.imread(mask01_root + mask10p)/255.
        h_mask, w_mask = mask.shape


        out_img, maskmap,maskmap_contour,mask_map_contour_grad,maskmap_contour_big \
            = multiply_mask(out_img, mask, mask10, mask01, h, w, h_mask, w_mask,
                            maskmap,maskmap_contour,mask_map_contour_grad,
                            maskmap_contour_big, if_bigMask, nums,area)

    _degra=out_img.astype('uint8')
    _degra_mask_grad= mask_map_contour_grad.astype('uint8')
    _degra_mask= (maskmap_contour*255.).astype('uint8')
    _degra_maskbig= (maskmap_contour_big*255.).astype('uint8')

    _degra_mask_r = np.ones_like(_degra_mask)*255. -_degra_mask
    _degra_maska = (np.add(_degra_mask_grad, _degra_mask_r)).astype('uint8')
    a = extractM.ExtractM(_degra)
    _degra_maskb = (extractM.Mask01_GradMask(a)).astype('uint8')

    return _degra, _degra_maska, _degra_maskb



if __name__=='__main__':
    folder='../simu_sp_data/'
    im='000.png'
    img=io.imread(folder+im)
    savefolder=folder

    mask_root = '../simu_sp_data/sp_mask_bank/mask/'
    mask10_root = '../simu_sp_data/sp_mask_bank/mask10/'
    mask01_root = '../simu_sp_data/sp_mask_bank/mask01/'

    _degra, _degra_maska, _degra_maskb=SimuSP(img, mask_root, mask10_root, mask01_root)

    io.imsave(savefolder+im.strip('.png')+'_degra.png',_degra)
    io.imsave(savefolder+im.strip('.png')+'_degra_maska.png',_degra_maska)
    io.imsave(savefolder+im.strip('.png')+'_degra_maskb.png',_degra_maskb)


