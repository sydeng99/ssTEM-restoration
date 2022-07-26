import cv2
import numpy as np
import os
import fill_contours
from skimage import io

def ExtractM(img, if_select_masks=True):
    if img.ndim==3:
        img=img[:,:,0]
    img_b=np.zeros([img.shape[0]+200,img.shape[1]+200],'uint8')
    img_b[:100]=img_b[-100:]=img_b[:,100:]=img_b[:,:-100]=255.
    img_b[100:-100,100:-100]=img

    gray = img_b
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy= cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    area=[]
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    area2=area
    area2=sorted(area)
    # print(len(area2))
    van = np.zeros_like(img_b, np.uint8)
    van2=np.ones_like(img_b,np.uint8)*255
    van_temp=np.zeros_like(img_b, np.uint8)
    for ii in range(len(area2)-1):
        if area2[ii]>50:
            current_area=area2[ii]
            contour_i=area.index(area2[ii])
            van_temp = np.zeros_like(img_b, np.uint8)
            x, y, w, h = cv2.boundingRect(contours[contour_i])
            cv2.drawContours(van_temp, contours, contour_i, (255, 255, 255), 0)
            cv2.fillPoly(van_temp, [contours[contour_i]], (255, 255, 255))

            mask01_temp=van_temp[y:y+h,x:x+w]
            img_temp=img_b[y:y+h,x:x+w]
            masked_img_temp=np.multiply((mask01_temp.astype('float')/255.), img_temp.astype('float'))
            # print(np.sum(masked_img_temp.astype('float')) / current_area)

            if if_select_masks:
                if np.sum(masked_img_temp.astype('float')) / current_area<=(220/3):
                    cv2.drawContours(van, contours, contour_i, (255, 255, 255), 10)
                    cv2.drawContours(van2, contours, contour_i, (0, 0, 0), 10)

                    cv2.fillPoly(van, [contours[contour_i]], (255, 255, 255))
                    cv2.fillPoly(van2,[contours[contour_i]], (0,0,0))

            else:
                cv2.drawContours(van, contours, contour_i, (255, 255, 255), 40)
                cv2.drawContours(van2, contours, contour_i, (0, 0, 0), 40)

                cv2.fillPoly(van, [contours[contour_i]], (255, 255, 255))
                cv2.fillPoly(van2, [contours[contour_i]], (0, 0, 0))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (70, 70))
    van2 = cv2.morphologyEx(van2, cv2.MORPH_CLOSE, kernel)
    van = (np.ones_like(van2)*255. - van2).astype('uint8')

    return van[100:-100,100:-100]


def Mask01_GradMask(mask10map):
    if mask10map.ndim==3:
        mask10map=mask10map[:,:,0]
    [h,w]=mask10map.shape
    mask10map_padding = np.zeros([h+100, w+100], 'uint8')
    mask10map_padding[50:-50, 50:-50]=mask10map
    ret, binary = cv2.threshold(mask10map_padding, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    area_ori_list = []
    for k in range(len(contours)):
        area_ori_list.append(cv2.contourArea(contours[k]))
    area2 = sorted(area_ori_list)

    selected_contour_id=[]

    for area in area2:
        if area>200:
            contour_i = area_ori_list.index(area)
            selected_contour_id.append(contour_i)

    im_e_copy = mask10map_padding.copy()
    im_e_copy2 = mask10map_padding.copy()
    maskmap10 = np.zeros_like(mask10map_padding)

    for ci in selected_contour_id:
        cv2.fillPoly(mask10map_padding, [contours[ci]], 255)
        cv2.drawContours(mask10map_padding, contours, ci, (255, 255, 255), thickness=1)

        x, y, w, h = cv2.boundingRect(contours[ci])
        mask10 = mask10map_padding[y:y + h, x:x + w]
        area= cv2.contourArea(contours[ci])

        scales = np.linspace(1.0, 0.05, 50)
        real_ave_alpha = fill_contours.AverageAlpha(area)
        if area>300000:
            area = 300000
            real_ave_alpha = fill_contours.AverageAlpha(area)
            values_array = fill_contours.DisValue_Unified2(scales, 300000, real_ave_alpha)
        else:
            values_array = fill_contours.DisValue_Unified2(scales, area, real_ave_alpha)
            
        values_array = values_array / np.max(values_array)

        blank_img = np.zeros_like(mask10).astype('float')
        for ss in range(len(scales) - 1):
            scale1 = scales[ss]
            scale2 = scales[ss + 1]

            white_ring = (fill_contours.scaled_contour_multiscale_simplify(mask10, scale1, scale2) / 255.).astype('float')

            if np.max(white_ring) > 0:
                present_value = values_array[ss]
                blank_img += (white_ring * present_value).astype('float')

        scale = scales[-1]
        white_center = fill_contours.scaled_center(mask10, scale).astype('float')

        assert values_array[-1] >= 0

        if values_array[-1] == 0:
            vvv = values_array[-2]
        else:
            vvv = values_array[-1]

        blank_img += (white_center / 255. * vvv).astype('float')

        all_1_mask = np.ones_like(blank_img)

        im_e_copy[y:y + h, x:x + w] = blank_img * 255.
        im_e_copy2[y:y + h, x:x + w] = (all_1_mask - blank_img) * 255.


    maskmap10=mask10map_padding
    maskmap01 = ((np.ones_like(maskmap10) * 255.).astype('float') - maskmap10.astype('float')).astype('uint8')
    grad_maskbig=im_e_copy+maskmap01
    grad_maskbig=grad_maskbig[50:-50,50:-50]
    return grad_maskbig

if __name__=='__main__':
    degra_img=io.imread('../simu_sp_data/008_degraB1.png')
    a=ExtractM(degra_img)
    io.imsave('../simu_sp_data/a.png', a) # Gen 01 Mask
    b = Mask01_GradMask(a)
    io.imsave('../simu_sp_data/b.png', b)  # Gen Grad Mask