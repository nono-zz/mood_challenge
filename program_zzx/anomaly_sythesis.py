import os
import numpy as np
import torch
import cv2
import random

from skimage.draw import random_shapes

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
        
        stripStart = random.randint(start, start + width)
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
        stripStart = random.randint(start, start + width)
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


"""random shape
"""
def randomShape(img):
    result, _ = random_shapes((128, 128), max_shapes=1, shape='rectangle', intensity_range=(0, 255), 
                       channel_axis=None, random_seed=0)
    
    print('done')
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    
    # method = blackStrip
    method = distortion
    method = randomShape
    
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
        
        try:
            new_img, gt_mask = method(img)
            
            
            cv2.imwrite(os.path.join(save_dir, f), new_img)
            cv2.imwrite(os.path.join(save_dir, f.replace('.png', '_gt.png')), gt_mask*255)
        except:
            new_img = method(img)
            cv2.imwrite(os.path.join(save_dir, f), new_img)
        
        