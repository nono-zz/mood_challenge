import os
import numpy as np
import torch
import cv2


"""Here we define all kinds of pseudo anomalies that can be directly apply on single images
"""


# corruptions
def blackStrip(img):
    size = 
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    
    method = blackStrip
    
    img_dir = '/home/zhaoxiang/mood_challenge/Sample_images/raw'
    save_dir = '/home/zhaoxiang/mood_challenge/Sample_images/{}'.format(str(method))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    for f in os.listdir(img_dir):
        img_path = os.path.join(img_dir, f)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        new_img = method(img)
        
        cv2.imwrite(f, new_img)
        
        