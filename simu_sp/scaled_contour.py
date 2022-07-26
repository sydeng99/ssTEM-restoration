import cv2
from skimage import io
import numpy as np
from PIL import Image

def scaled_center(maskImg,scale,if_show=False):
    if maskImg.ndim==3:
        gray = cv2.cvtColor(maskImg, cv2.COLOR_BGR2GRAY)
    else:
        gray=maskImg
    ret, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    area2 = sorted(area)
    contour_i = area.index(max(area2))

    img_wb = np.zeros_like(maskImg)
    cv2.fillPoly(img_wb, [contours[contour_i]], (255, 255, 255))
    scaled_img_wb = cv2.resize(img_wb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if if_show:
        cv2.imshow('1',img_wb)
        cv2.waitKey(0)
        cv2.imshow('1',scaled_img_wb)
        cv2.waitKey(0)

    M = cv2.moments(contours[contour_i]) 
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cxs=int(cx*scale)
    cys=int(cy*scale)
    if scaled_img_wb.ndim==3:
        [h, w, c] = scaled_img_wb.shape
    else:
        [h,w]=scaled_img_wb.shape
    img_b = np.zeros_like(maskImg)
    img_b[cy - cys:cy - cys + h, cx - cxs:cx - cxs + w] = scaled_img_wb
    return img_b


def scaled_contour(maskImg,scale):
    if maskImg.ndim==3:
        gray = cv2.cvtColor(maskImg, cv2.COLOR_BGR2GRAY)
    else:
        gray=maskImg
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    img_wb = np.zeros_like(maskImg)
    cv2.drawContours(img_wb, contours, -1, (255, 255, 255))
    cv2.fillPoly(img_wb, [contours[0]], (255, 255, 255))
    M = cv2.moments(contours[0]) 
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    img_b = np.zeros_like(maskImg)
    scaled_img_wb = cv2.resize(img_wb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if scaled_img_wb.ndim==3:
        gray = cv2.cvtColor(scaled_img_wb, cv2.COLOR_BGR2GRAY)
    else:
        gray=scaled_img_wb
    rets, binarys = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours_scaled, hierarchy_scaled = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(scaled_img_wb, contours_scaled, -1, (255, 255, 255))
    
    Ms = cv2.moments(contours_scaled[0])
    if M['m00']!=0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx = int((M['m10']+0.01) / (M['m00']+0.01))
        cy = int((M['m01']+0.01) / (M['m00']+0.01))
        
    if scaled_img_wb.ndim==3:
        [h, w, c] = scaled_img_wb.shape
    else:
        [h,w]=scaled_img_wb.shape
    cxs=int(cx*scale)
    cys=int(cy*scale)
    img_b[cy - cys:cy - cys + h, cx - cxs:cx - cxs + w] = scaled_img_wb
    img_bw = 255 - img_b
    
    img = (np.multiply(img_wb / 255., img_bw / 255.) * 255.).astype('uint8')
    return img


def scaled_contour_multiscale(maskImg,scale1,scale2):
    if maskImg.ndim==3:
        gray = cv2.cvtColor(maskImg, cv2.COLOR_BGR2GRAY)
    else:
        gray=maskImg
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    img_wb = np.zeros_like(maskImg)
    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    area2=area
    area2 = sorted(area)
    contour_i = area.index(max(area2))

    cv2.fillPoly(img_wb, [contours[contour_i]], (255, 255, 255))
    M = cv2.moments(contours[contour_i])
    if M['m00']!=0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx = int((M['m10']+0.01) / (M['m00']+0.01))
        cy = int((M['m01']+0.01) / (M['m00']+0.01))

    img_b0 = np.zeros_like(maskImg)
    scaled_img_wb = cv2.resize(img_wb, None, fx=scale1, fy=scale1, interpolation=cv2.INTER_CUBIC)
    if scaled_img_wb.ndim==3:
        gray = cv2.cvtColor(scaled_img_wb, cv2.COLOR_BGR2GRAY)
    else:
        gray=scaled_img_wb
    rets, binarys = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours_scaled, hierarchy_scaled = cv2.findContours(binarys, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(scaled_img_wb, contours_scaled, -1, (255, 255, 255))
    cxs=int(cx*scale1)
    cys=int(cy*scale1)
    
    if scaled_img_wb.ndim==3:
        [h, w, c] = scaled_img_wb.shape
    else:
        [h,w]=scaled_img_wb.shape
    if img_b0[cy - cys:cy - cys + h, cx - cxs:cx - cxs + w].shape == scaled_img_wb.shape:
        img_b0[cy - cys:cy - cys + h, cx - cxs:cx - cxs + w] = scaled_img_wb

        img_b = np.zeros_like(maskImg)
        scaled_img_wb = cv2.resize(img_wb, None, fx=scale2, fy=scale2, interpolation=cv2.INTER_CUBIC)
        if scaled_img_wb.ndim==3:
            gray = cv2.cvtColor(scaled_img_wb, cv2.COLOR_BGR2GRAY)
        else:
            gray=scaled_img_wb
        rets, binarys = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours_scaled, hierarchy_scaled = cv2.findContours(binarys, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(scaled_img_wb, contours_scaled, -1, (255, 255, 255))
        
        cxs=int(cx*scale2)
        cys=int(cy*scale2)
        
        if scaled_img_wb.ndim==3:
            [h, w, c] = scaled_img_wb.shape
        else:
            [h,w]=scaled_img_wb.shape
        if img_b[cy - cys:cy - cys + h, cx - cxs:cx - cxs + w].shape==scaled_img_wb.shape:
            img_b[cy - cys:cy - cys + h, cx - cxs:cx - cxs + w] = scaled_img_wb
            
            img_bw = 255 - img_b

            img = (np.multiply(img_b0 / 255., img_bw / 255.) * 255.).astype('float')
            return img
        else:
            return np.zeros_like(scaled_img_wb)

    else:
        return np.zeros_like(scaled_img_wb)



def scaled_contour_multiscale_simplify(maskImg,scale1,scale2):
    if maskImg.ndim==3:
        gray = cv2.cvtColor(maskImg, cv2.COLOR_BGR2GRAY)
    else:
        gray=maskImg
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img_wb = np.zeros_like(maskImg)

    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    area2 = sorted(area)
    contour_i = area.index(max(area2))

    cv2.fillPoly(img_wb, [contours[contour_i]], (255, 255, 255))
    M = cv2.moments(contours[contour_i])
    if M['m00']!=0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx = int((M['m10']+0.01) / (M['m00']+0.01))
        cy = int((M['m01']+0.01) / (M['m00']+0.01))


    img_b0 = np.zeros_like(maskImg)
    scaled_img_wb = cv2.resize(img_wb, None, fx=scale1, fy=scale1, interpolation=cv2.INTER_CUBIC)

    cxs=int(cx*scale1)
    cys=int(cy*scale1)

    if scaled_img_wb.ndim==3:
        [h, w, c] = scaled_img_wb.shape
    else:
        [h,w]=scaled_img_wb.shape
    if img_b0[cy - cys:cy - cys + h, cx - cxs:cx - cxs + w].shape == scaled_img_wb.shape:
        img_b0[cy - cys:cy - cys + h, cx - cxs:cx - cxs + w] = scaled_img_wb

        img_b = np.zeros_like(maskImg)
        scaled_img_wb = cv2.resize(img_wb, None, fx=scale2, fy=scale2, interpolation=cv2.INTER_CUBIC)

        cxs=int(cx*scale2)
        cys=int(cy*scale2)

        if scaled_img_wb.ndim==3:
            [h, w, c] = scaled_img_wb.shape
        else:
            [h,w]=scaled_img_wb.shape
        if img_b[cy - cys:cy - cys + h, cx - cxs:cx - cxs + w].shape==scaled_img_wb.shape:
            img_b[cy - cys:cy - cys + h, cx - cxs:cx - cxs + w] = scaled_img_wb

            img_bw = 255 - img_b
            img = (np.multiply(img_b0 / 255., img_bw / 255.) * 255.).astype('float')
            return img
        else:
            return np.zeros_like(scaled_img_wb)

    else:
        return np.zeros_like(scaled_img_wb)


def refine_contour_multiscale(maskImg,scale1,scale2):
    if maskImg.ndim==3:
        gray = cv2.cvtColor(maskImg, cv2.COLOR_BGR2GRAY)
    else:
        gray=maskImg
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img_wb = np.zeros_like(maskImg)
    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    area2=area
    area2 = sorted(area)
    contour_i = area.index(max(area2))

    cv2.fillPoly(img_wb, [contours[contour_i]], (0, 0, 0))
    M = cv2.moments(contours[contour_i])
    if M['m00']!=0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx = int((M['m10']+0.01) / (M['m00']+0.01))
        cy = int((M['m01']+0.01) / (M['m00']+0.01))

    img_b0 = np.zeros_like(maskImg)
    scaled_img_wb = cv2.resize(img_wb, None, fx=scale1, fy=scale1, interpolation=cv2.INTER_CUBIC)
    if scaled_img_wb.ndim==3:
        gray = cv2.cvtColor(scaled_img_wb, cv2.COLOR_BGR2GRAY)
    else:
        gray=scaled_img_wb
    rets, binarys = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours_scaled, hierarchy_scaled = cv2.findContours(binarys, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(scaled_img_wb, contours_scaled, -1, (0, 0, 0))
    cxs=int(cx*scale1)
    cys=int(cy*scale1)
    if scaled_img_wb.ndim==3:
        [h, w, c] = scaled_img_wb.shape
    else:
        [h,w]=scaled_img_wb.shape
    if img_b0[cy - cys:cy - cys + h, cx - cxs:cx - cxs + w].shape == scaled_img_wb.shape:
        img_b0[cy - cys:cy - cys + h, cx - cxs:cx - cxs + w] = scaled_img_wb
        
        img_b = np.zeros_like(maskImg)
        scaled_img_wb = cv2.resize(img_wb, None, fx=scale2, fy=scale2, interpolation=cv2.INTER_CUBIC)
        if scaled_img_wb.ndim==3:
            gray = cv2.cvtColor(scaled_img_wb, cv2.COLOR_BGR2GRAY)
        else:
            gray=scaled_img_wb
        rets, binarys = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours_scaled, hierarchy_scaled = cv2.findContours(binarys, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(scaled_img_wb, contours_scaled, -1, (0, 0, 0))
        cxs=int(cx*scale2)
        cys=int(cy*scale2)
        if scaled_img_wb.ndim==3:
            [h, w, c] = scaled_img_wb.shape
        else:
            [h,w]=scaled_img_wb.shape
        if img_b[cy - cys:cy - cys + h, cx - cxs:cx - cxs + w].shape==scaled_img_wb.shape:
            img_b[cy - cys:cy - cys + h, cx - cxs:cx - cxs + w] = scaled_img_wb
            img_bw = 255 - img_b
            img = (np.multiply(img_b0 / 255., img_bw / 255.) * 255.).astype('uint8')
            return img
        else:
            return np.zeros_like(scaled_img_wb)

    else:
        return np.zeros_like(scaled_img_wb)
