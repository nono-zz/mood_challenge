import cv2
import torch
import numpy as np
# img =  cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)
# print('done')




"""generate gaussian noise and check if it's continous
"""

ns = torch.normal(mean=torch.zeros(5, 512, 512), std=0.2)
ns = (ns + 1)/2 * 255
ns_arr = ns.numpy().astype(np.uint8)


for i in range(ns.shape[0]):
    cv2.imwrite('gaussian_noise_{}.png'.format(i), ns_arr[i,:,:])