import os
import numpy as np
import torch
import cv2
import random
from PIL import Image, ImageOps
from skimage.draw import random_shapes

from cutpaste_sythesis import CutPasteUnion, CutPaste3Way

import skimage.exposure
import numpy as np
from numpy.random import default_rng

# from scipy.misc import lena


"""Here we define all kinds of pseudo anomalies that can be directly apply on single images
"""

def getBbox(image):
    mask = np.zeros_like(image)
    B = np.argwhere(image)
    (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
    mask[ystart:ystop, xstart:xstop] = 1
    return mask, (ystart, xstart), (ystop, xstop)
        

# corruptions
def singleStrip(img, start, stop, mode, p = 0.3):
    if mode == 0:       # do horizonally
        start = start[0]
        stop = stop[0]
        # width = random.randint(0, start - stop)
        width = random.randint(0, int((stop - start) * p))
        
        stripStart = random.randint(start, stop - width)
        stripStop = stripStart + width
        
        # generate a mask
        mask = np.ones_like(img)
        mask[stripStart:stripStop, :] = 0
        
        new_img = mask * img
        return new_img, mask
    
    elif mode == 1:
        start = start[1]
        stop = stop[1]
        # width = random.randint(start, stop)
        
        width = random.randint(0, int((stop - start) * p))
        stripStart = random.randint(start, stop - width)
        stripStop = stripStart + width
        
        # generate a mask
        mask = np.ones_like(img)
        mask[:, stripStart:stripStop] = 0
        
        new_img = mask * img
        return new_img, mask
        


def blackStrip(img):
    # try:
    mask, start, stop = getBbox(img)
        
    # except:
    #     return None
    if mask.sum() > 800:
        # decide which mode it is
        mode = random.randint(0,2)
        if mode != 2:       # do horizonally
            new_img, stripMask = singleStrip(img, start, stop, mode)
            gtMask = mask*(1-stripMask)
            # return new_img, gtMask
            return new_img        
        else:
            img_1, stripMask_1 = singleStrip(img, start, stop, mode = 0)
            new_img, stripMask_2 = singleStrip(img_1, start, stop, mode = 1)

            gt_mask = (1 - (stripMask_1 * stripMask_2)) * mask
            # return new_img, gt_mask
            return new_img
        
    else:
        return img

            
        
        
""" distortion
"""
def distortion(sss):
    img = sss
    symbol = random.randint(0,1)
    if symbol == 0:
        A = img.shape[0] / 3.0
    else:
        A = -img.shape[0] / 3.0
    
    i = random.randint(3,7)
    w = i/100 / img.shape[1]

    shift = lambda x: A * np.sin(2.0*np.pi*x * w)

    mode = random.randint(0,2)
    if mode == 0:
        for i in range(img.shape[0]):
            img[:,i] = np.roll(img[:,i], int(shift(i)))
    elif mode == 1:
        for i in range(img.shape[0]):
            img[i,:] = np.roll(img[i,:], int(shift(i)))
    else:
        for i in range(img.shape[0]):
            img[:,i] = np.roll(img[:,i], int(shift(i)))
        for i in range(img.shape[0]):
            img[i,:] = np.roll(img[i,:], int(shift(i)))
    return img


def cp(img_path):
    img = Image.open(img_path)
    img = ImageOps.grayscale(img)
    
    org, cut_img = cutpaste(img)
    return org, cut_img


"""random shape
"""
def randomShape(img, scaleUpper=255):


    # define random seed to change the pattern
    rng = default_rng()

    # define image size
    width=img.shape[0]
    height=img.shape[1]

    # create random noise image
    noise = rng.integers(0, 255, (height,width), np.uint8, True)

    # blur the noise image to control the size
    blur = cv2.GaussianBlur(noise, (0,0), sigmaX=15, sigmaY=15, borderType = cv2.BORDER_DEFAULT)

    # stretch the blurred image to full dynamic range
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0,255)).astype(np.uint8)

    # threshold stretched image to control the size
    thresh = cv2.threshold(stretch, 200, 255, cv2.THRESH_BINARY)[1]

    # apply morphology open and close to smooth out shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    result = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    mask, start, stop = getBbox(img)
    anomalyMask = mask * result
    anomalyMask = np.where(anomalyMask > 0, 1, 0)
    
    addImg = np.ones_like(img)
    scale = random.randint(0,scaleUpper)
    
    augImg = img * (1-anomalyMask) + addImg * anomalyMask * scale
    return augImg.astype(np.uint8)
    
    

    # # save result
    # cv2.imwrite('random_blobs.png', result)

    # # show results
    # # cv2.imshow('noise', noise)
    # cv2.imwrite('noise.png', noise)
    # cv2.imwrite('blur.png', blur)
    # cv2.imwrite('stretch.png', stretch)
    # cv2.imwrite('thresh.png', thresh)
    # cv2.imwrite('result.png', result)
    
    # cv2.imshow('blur', blur)
    # cv2.imshow('stretch', stretch)
    # cv2.imshow('thresh', thresh)
    # cv2.imshow('result', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
        
    
    
    

    
    
    
    
    
    
    
if __name__ == '__main__':
    
    # method = blackStrip
    method = distortion
    method = randomShape
    
    # method = cp
    cutpaste = CutPasteUnion(transform=None)
    
    img_dir = '/home/zhaoxiang/mood_challenge/Sample_images/raw'
    save_dir = '/home/zhaoxiang/mood_challenge/Sample_images/{}'.format(method.__name__)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
        
    files = os.listdir(img_dir)
    files.sort()
    for f in files:
        img_path = os.path.join(img_dir, f)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        
        # mask = np.where(img>0, 255, 0)
        # mask = mask.astype(np.uint8)
        # kernel = np.ones((5, 5), np.uint8)
        # mask_dil = cv2.dilate(mask, kernel, iterations=1)
        # cv2.imwrite('mask.png',mask_dil)
        
        if img.max() == 0:
            continue
        """cutpaste"""
        # org, cut_img = cp(img_path)
        # cut_img = cut_img.save(os.path.join(save_dir, f))
        
        
        """other augmentations"""
        try:
            new_img, gt_mask = method(img)
            
            
            cv2.imwrite(os.path.join(save_dir, f), new_img)
            cv2.imwrite(os.path.join(save_dir, f.replace('.png', '_gt.png')), gt_mask*255)
        except:
            new_img = method(img)
            cv2.imwrite(os.path.join(save_dir, f), new_img)
        
        