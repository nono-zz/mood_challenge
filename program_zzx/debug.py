import cv2
import torch
import numpy as np
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
# img =  cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)
# print('done')




"""generate gaussian noise and check if it's continous
"""

# ns = torch.normal(mean=torch.zeros(5, 512, 512), std=0.2)
# ns = (ns + 1)/2 * 255
# ns_arr = ns.numpy().astype(np.uint8)


# for i in range(ns.shape[0]):
#     cv2.imwrite('gaussian_noise_{}.png'.format(i), ns_arr[i,:,:])




"""test the grayscale of the imaged saved to slice
"""
img = nib.load('/home/zhaoxiang/mood_challenge_data/data/brain/brain_train/00000.nii.gz')

nimg_array = img.get_fdata()  

slice = nimg_array[:,:,95]  
print(slice.max())

matplotlib.image.imsave('sample.png', slice, cmap='gray')

img_sample = cv2.imread('/home/zhaoxiang/sample.png', cv2.IMREAD_GRAYSCALE)
print(img_sample.maxz())

img = cv2.imread('/home/zhaoxiang/dataset/Mood_brain/train/good/00000_95_raw.png', cv2.IMREAD_GRAYSCALE)
print(img.max())

