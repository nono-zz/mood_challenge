# requirement: install nibabel
import os
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import cv2
import numpy as np


def check_files():

    # only extract images from the training set.


    # dir_path = '/home/zhaoxiang/mood_challenge_data/brain_train'
    # dir_path = '/home/zhaoxiang/mood_challenge_data/data/brain/toy'
    # dir_path = '/home/zhaoxiang/mood_challenge_data/data/brain/toy_label/pixel'
    dir_path = '/home/zhaoxiang/mood_challenge_data/data/brain/train'
    filenames = os.listdir(dir_path)
    filenames.sort()
    for file in filenames:
        if file.startswith('.'):
            continue
        file_path = os.path.join(dir_path, file)
        img = nib.load(file_path)          # read nii
        img_fdata = img.get_fdata()
        fname = file.replace('.nii.gz', '')
        if 'label' in dir_path:
            img_f_path = os.path.join('/home/zhaoxiang/mood_challenge_data', fname)
            raw_path = os.path.join(img_f_path, 'gt')
            a = 'gt'
        else:
            img_f_path = os.path.join('/home/zhaoxiang/dataset/Mood_brain', fname)
            raw_path = os.path.join(img_f_path, 'raw')  
            a = 'raw' 
            
        if not os.path.exists(img_f_path):
            os.mkdir(img_f_path)
            
        if not os.path.exists(raw_path):
            os.mkdir(raw_path)
            
            
        (x,y,z) = img.shape
        for i in range(z):
            gray = img_fdata[:,:,i]
            # if np.max(gray)!=0:
            #     gray = np.where(gray == 6, 1, 0)
            matplotlib.image.imsave(os.path.join(raw_path, '{}_'.format(i) + a + '.png'), gray, cmap='gray')
        print(fname, '   done')



def save_files():
    # dir_path = '/home/zhaoxiang/mood_challenge_data/data/brain/toy_label/pixel'
    dir_path = '/home/zhaoxiang/mood_challenge_data/data/brain/toy'
    # dir_path = '/home/zhaoxiang/mood_challenge_data/data/brain/val'
    filenames = os.listdir(dir_path)
    filenames.sort()
    for file in filenames:
        if file.startswith('.'):
            continue
        file_path = os.path.join(dir_path, file)
        img = nib.load(file_path)          # read nii
        img_fdata = img.get_fdata()
        fname = file.replace('.nii.gz', '')
        if 'label' in dir_path:
            raw_path = os.path.join('/home/zhaoxiang/dataset/Mood_brain_cv2', 'test_label')
            a = 'gt'
        else:
            raw_path = os.path.join('/home/zhaoxiang/dataset/Mood_brain_cv2', 'test')  
            a = 'raw'
            
        if not os.path.exists(raw_path):
            os.mkdir(raw_path)
            
        (x,y,z) = img.shape
        for i in range(z):
            gray = img_fdata[:,:,i]
            # if np.max(gray)!=0:
            #     gray = np.where(gray == 6, 1, 0)
            # matplotlib.image.imsave(os.path.join(raw_path, '{}_{}_'.format(fname, i) + a + '.png'), gray, cmap='gray')
            cv2.imwrite(os.path.join(raw_path, '{}_{}_'.format(fname, i) + a + '.png'), (gray*255).astype(np.uint8))
            
        print(fname, '   done')


save_files()