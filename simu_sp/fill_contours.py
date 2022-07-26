from skimage import io
import numpy as np
import os
import cv2
import pandas as pd
import math
from scipy.ndimage import binary_erosion
from scaled_contour import scaled_contour_multiscale,scaled_center,refine_contour_multiscale,scaled_contour_multiscale_simplify


def Opening(mask):
    mask=mask.astype('float')
    k = np.ones((3, 3), np.uint8)
    img_opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=k, iterations=1)
    img_opening=img_opening.astype('uint8')
    return img_opening


def AreaScales_unified(area):
    if area <= 5000:
        scales=np.linspace(1.0,0.05,20)
    elif area > 5000 and area <= 15000:
        scales=np.linspace(1.0,0.05,25)
    elif area>15000 and area<=30000:
        scales=np.linspace(1.0,0.05,35)
    elif area > 30000 and area <= 80000:
        scales = np.linspace(1.0, 0.05, 50)
    else:
        scales=np.linspace(1.0,0.05,70)
    return scales



def AverageAlpha(area):
    ap1 = -9.662 * math.pow(10, -17)
    ap2 = 4.709 * math.pow(10, -11)
    ap3 = -7.72 * math.pow(10, -6)
    ap4 = 0.7038

    ave_alpha = ap1*math.pow(area,3)+ap2*math.pow(area,2)+ap3*area+ap4
    return ave_alpha


def DisValue_Unified2(scale,area,average_intensity):
    if area <= 10000:
        p1=0.04202
        p2=0.5332
        p3=-0.02127
        p4=0.7099
    elif area > 10000 and area<=60000:
        p1=0.7586
        p2=0.2427
        p3=0.2224
        p4=0.386
    elif area >60000:
        p1=2.104
        p2= -0.4489
        p3=0.4921
        p4=-0.02028
    else:
        raise NotImplementedError

    disvalue = p1 * np.power(scale, 3) + p2 * np.power(scale, 2) + p3 * scale + p4
    disvalue = disvalue * average_intensity
    return disvalue



def Fill_alphaMap(mask_contour, enlarge_scale=1.2):
    mask10 = io.imread(mask_contour)
    area=(np.sum((mask10/255.)).astype('float'))
    area=area*enlarge_scale
    if area>0:
        mask10=cv2.resize(mask10, None, fx=enlarge_scale, fy=enlarge_scale, interpolation=cv2.INTER_CUBIC)
        scales=AreaScales_unified(area)
        real_ave_alpha=AverageAlpha(area)
        values_array = DisValue_Unified2(scales, area, real_ave_alpha)

        values_array=values_array/np.max(values_array)

        blank_img = np.zeros_like(mask10).astype('float')
        for ss in range(len(scales) - 1):
            scale1 = scales[ss]
            scale2 = scales[ss + 1]

            white_ring = (scaled_contour_multiscale_simplify(mask10, scale1, scale2) / 255.).astype('float')

            if np.max(white_ring) > 0:
                present_value = values_array[ss]
                blank_img += (white_ring * present_value).astype('float')

        scale = scales[-1]
        white_center = scaled_center(mask10, scale).astype('float')

        assert values_array[-1] >= 0

        if values_array[-1] == 0:
            vvv = values_array[-2]
        else:
            vvv = values_array[-1]

        blank_img += (white_center / 255. * vvv).astype('float')

        return (blank_img*255.).astype('uint8')



def SaveMask01(mask_path,savepath,savepath10,if_show=False):
    masks=os.listdir(mask_path)
    for mask in masks:
        mask_img=io.imread(os.path.join(mask_path,mask))

        gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        area = []
        for k in range(len(contours)):
            area.append(cv2.contourArea(contours[k]))
        area2 = sorted(area)
        print(area2)

        contour_i = area.index(area2[-1])
        if if_show:
            cv2.drawContours(mask_img, contours, contour_i, (0, 0, 255), thickness=1)
            cv2.imshow('1',mask_img)
            cv2.waitKey(0)

        van=np.zeros_like(mask_img,'uint8')
        cv2.fillPoly(van, [contours[contour_i]], (255, 255, 255))
        # cv2.drawContours(van, contours, contour_i, (0, 0, 0), 1)

        if not os.path.exists(savepath):
            os.makedirs(savepath)
        io.imsave(savepath+mask,van)

        van10=255-van
        if not os.path.exists(savepath10):
            os.makedirs(savepath10)
        io.imsave(savepath10+mask,van10)

def DisTransAlpha(out_mask):
    out_mask = binary_erosion(out_mask, iterations=10, border_value=1)

    out_mask = (out_mask * 255).astype(np.uint8)
    return out_mask


if __name__ =='__main__':
    folder = '../simu_sp_data/contour/'
    savefolder = '../simu_sp_data/mask/'
    savefolder10 = '../simu_sp_data/mask10/'
    savefolder01 = '../simu_sp_data/mask01/'
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    if not os.path.exists(savefolder10):
        os.makedirs(savefolder10)
    if not os.path.exists(savefolder01):
        os.makedirs(savefolder01)
    mask_contour=os.path.join(folder, 'contour.png')
    mask = Fill_alphaMap(mask_contour)
    io.imsave(os.path.join(savefolder, 'MaskFill_contour.png'),mask)
    SaveMask01(savefolder, savefolder01, savefolder10)