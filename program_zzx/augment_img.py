from anomaly_sythesis import blackStrip, distortion
import os
import cv2
import numpy as np

method = blackStrip

mode = 'train'

if mode == 'train':
    
    dir_path = '/home/zhaoxiang/dataset/Mood_brain_cv2/{}/good'.format(mode)
else:
    dir_path = '/home/zhaoxiang/dataset/Mood_brain_cv2/{}'.format(mode)
    
files = os.listdir(dir_path)
files.sort()

save_dir = dir_path.replace(mode, '{}_{}'.format(mode, method.__name__))
if not os.path.exists(save_dir):
    # os.mkdir(save_dir,)
    os.makedirs(save_dir)
    
for filename in files:
    file_path = os.path.join(dir_path, filename)
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  
    
    if img.max() == 0:
            continue
        
    try:
        new_img, gt_mask = method(img)
        
        
        cv2.imwrite(os.path.join(save_dir, filename.replace('.png', '_strip.png')), new_img)
        # cv2.imwrite(os.path.join(save_dir, filename.replace('.png', '_gt.png')), gt_mask*255)
    except:
        # new_img = method(img)
        # cv2.imwrite(os.path.join(save_dir, filename), new_img)
        continue
    
