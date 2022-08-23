from ctypes import sizeof
from logging import exception
import os
from this import d
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
import random
import torch.nn.functional as F
import torchvision.transforms as transforms
import nibabel as nib
import torch.nn
import random
from perlin import rand_perlin_2d_np

from torchvision.utils import save_image    

import torchio as tio


class TestDataset(Dataset):

    def __init__(self, root_dir, mode='brain', anomaly_source_path=None, size=[256,256], augumentation=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=size
        self.augumentation=augumentation
        self.dispatcher = {'gaussianSeperate': self.gaussianSeperate,
                           'gaussianUnified': self.gaussianUnified,
                           'Circle': self.Circle_Aug}

        self.img_dir = os.path.join(root_dir, mode)
        # self.image_paths = sorted(glob.glob(self.img_dir+"/{}_train".format(mode)+"/*.nii.gz"))
        self.image_paths = sorted(glob.glob(self.img_dir+"/toy"+"/*.nii.gz"))
        
        
        if anomaly_source_path:
            self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                    #   iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                    #   iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        
        self.transform =  transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.Resize((size, size)),
                                            transforms.ToTensor()])
        

    def __len__(self):
        return len(self.image_paths)


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug
    
    def getBbox(self, image):
        image = np.squeeze(image, axis=2)
        mask = np.zeros_like(image)
        B = np.argwhere(image)
        try:
            (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
            mask[ystart:ystop, xstart:xstop] = 1
        except:
            return np.expand_dims(image, axis=2)
            
        mask = np.expand_dims(image, axis=2)
        return mask
        

    def DRAEM_Aug(self, image, anomaly_source_image):
        aug = self.randAugmenter()
        perlin_scale = 6                        # maybe determine the noise size
        min_perlin_scale = 0
        # anomaly_source_img = self.read_nib_2_cv2(anomaly_source_path)
        # anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_image)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            # objectMask = image > 0.01
            
            ObjectMask = self.getBbox(image)
            NoiseMask = (perlin_thr).astype(np.float32)
            msk = NoiseMask * ObjectMask
            augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)
    
    def gaussianSeperate(self, x, noise_res=16, noise_std=0.2):
        
        ns = torch.normal(mean=torch.zeros(x.shape[0], noise_res, noise_res), std=noise_std).to(x.device)

        ns = F.upsample_bilinear(ns, size=[x.shape[1], x.shape[2]])

        # Roll to randomly translate the generated noise.
        roll_x = random.choice(range(x.shape[1]))
        roll_y = random.choice(range(x.shape[2]))
        ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])

        # mask = x.sum(dim=1, keepdim=True) > 0.01
        mask = x > 0.01
        ns *= mask # Only apply the noise in the foreground.
        res = x + ns

        return res  
        
    def gaussianUnified(self, x, noise_res=16, noise_std=0.2):
            
        ns = torch.normal(mean=torch.zeros(1, 1, noise_res, noise_res), std=noise_std).to(x.device)

        ns = F.upsample_bilinear(ns, size=[x.shape[1], x.shape[2]])

        # Roll to randomly translate the generated noise.
        roll_x = random.choice(range(x.shape[1]))
        roll_y = random.choice(range(x.shape[2]))
        ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])
        
        
        size_to_slice = random.randint(2, x.shape[0])
        startID = random.randint(0, x.shape[0]-size_to_slice)
        Coef_ascend = np.arange(0,1,1 / ((size_to_slice)//2))
        Coef_descend = np.arange(1,0, -1 / (size_to_slice - size_to_slice//2))
        Coefs = np.zeros(x.shape[0])
        Coefs[startID:startID + size_to_slice//2] = Coef_ascend[:size_to_slice//2]
        Coefs[startID + size_to_slice//2 : size_to_slice + startID] = Coef_descend[:size_to_slice - size_to_slice//2]
        Coefs = Coefs.reshape(-1, 1, 1)           # [256, 1, 1]
        
        # stack the ns for x.shape[0] times
        ns = torch.squeeze(ns, dim=0)
        ns = ns.repeat(x.shape[0], 1, 1)

        # mask = x.sum(dim=1, keepdim=True) > 0.01
        mask = x > 0.01
        ns *= mask # Only apply the noise in the foreground.
        res = x + ns * Coefs

        return res  
    
    def Circle_Aug(x):
        from skimage import draw
        arr = np.zeros((200, 200))
        rr, cc = draw.circle_perimeter(100, 100, radius=80, shape=arr.shape)
        arr[rr, cc] = 1
        
    def read_nib_2_cv2(self, img_path):
        nimg = nib.load(img_path)
        nimg_array = nimg.get_fdata()           # 3d image
        img_list = []
        for i in range(nimg_array.shape[2]):
            slice = nimg_array[:,:,i]
            slice_arr = (slice*255).astype(np.uint8)
            # slice.arr = np.expand_dims(slice_arr, axis=0)
            img_list.append(slice_arr)
            
        img_arr = np.stack(img_list, axis=0)
        return img_arr
        
        

    def DRAEM_transform(self, image_path, anomaly_source_path):
        # read nibal as cv2 format
        object_3D = self.read_nib_2_cv2(image_path)
        object_3D = cv2.resize(object_3D, dsize=(self.resize_shape[1], self.resize_shape[0]))
        
        anomlaySource_3D = self.read_nib_2_cv2(anomaly_source_path)
        anomlaySource_3D = cv2.resize(anomlaySource_3D, dsize=(self.resize_shape[1], self.resize_shape[0]))

        # do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        # if do_aug_orig:
        #     object_3D = self.rot(object_3D=object_3D)
        
        image_list, augmented_image_list, anomaly_mask_list, has_anomaly_list = [], [], [], []
        for i, image in enumerate(object_3D):
            image = np.expand_dims(image, axis = 2)
            image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
            augmented_image, anomaly_mask, has_anomaly = self.DRAEM_Aug(image, np.expand_dims(anomlaySource_3D[i], axis = 2))
            augmented_image = np.transpose(augmented_image, (2, 0, 1))
            image = np.transpose(image, (2, 0, 1))
            anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
            
            image_list.append(image)
            augmented_image_list.append(augmented_image)
            anomaly_mask_list.append(anomaly_mask)
            has_anomaly_list.append(has_anomaly)
            
        image = np.stack(image_list).squeeze(axis=1)
        augmented_image = np.stack(augmented_image_list, axis=0).squeeze(axis=1)
        anomaly_mask = np.stack(anomaly_mask_list, axis=0).squeeze(axis=1)
        has_anomaly = np.stack(has_anomaly_list, axis=0).squeeze(axis=1)       
        
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        # if self.augumentation == 'DRAEM':
        #     anomaly_source_idx = torch.randint(0, len(self.image_paths), (1,)).item()
        #     image, augmented_image, anomaly_mask, has_anomaly = self.DRAEM_transform(self.image_paths[idx],
        #                                                                     self.image_paths[anomaly_source_idx])
        #     # sample = {'image': image, "anomaly_mask": anomaly_mask,
        #     #         'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}
            
        #     return image, anomaly_mask, augmented_image, has_anomaly
            
        
        # elif 'Gaussian' in self.augumentation:
            
        img_path = self.image_paths[idx]
        nimg = nib.load(img_path)
        nimg_array = nimg.get_fdata()           # 3d image
        img_list = []
        for i in range(nimg_array.shape[2]):
            # slice =  np.expand_dims(nimg_array[:,:,i], axis=0)
            slice = nimg_array[:,:,i]
            # nimg_tensor = self.transform(slice)
            nimg_tensor = torch.tensor(slice)
            nimg_tensor = torch.unsqueeze(nimg_tensor, dim = 0)
            img_list.append(nimg_tensor)
            save_image(nimg_tensor, 'data_loader_image.png')
            
        # img_tensor = torch.tensor(img_list)
        img_tensor = torch.cat(img_list, dim = 0)
        img_tensor = img_tensor.float()             # torch.tensor([256, 256, 256])
        
        # aug_tensor = self.dispatcher[self.augumentation](img_tensor)
        # aug_tensor = aug_tensor.float()
        

        return (img_tensor, img_path)




class TrainDataset(Dataset):

    def __init__(self, root_dir, mode='brain', anomaly_source_path=None, size=None, augumentation=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=size
        self.augumentation=augumentation
        self.dispatcher = {'gaussianSeperate': self.gaussianSeperate,
                           'gaussianUnified': self.gaussianUnified,
                           'Circle': self.Circle_Aug}

        self.img_dir = os.path.join(root_dir, mode)
        # self.image_paths = sorted(glob.glob(self.img_dir+"/{}_train".format(mode)+"/*.nii.gz"))
        self.image_paths = sorted(glob.glob(self.img_dir+"/train"+"/*.nii.gz"))
        
        
        if anomaly_source_path:
            self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                    #   iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                    #   iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        
        self.transform =  transforms.Compose([
                                            transforms.ToPILImage(),
                                            transforms.Resize((size, size)),
                                            transforms.ToTensor()])
        

    def __len__(self):
        return len(self.image_paths)


    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug
    
    def getBbox(self, image):
        image = np.squeeze(image, axis=2)
        mask = np.zeros_like(image)
        B = np.argwhere(image)
        try:
            (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
            mask[ystart:ystop, xstart:xstop] = 1
            mask = np.expand_dims(mask, axis=2)
            return mask
        except:
            return np.expand_dims(image, axis=2)
        
    def getBbox_3D(self, image):
        mask_3D = np.zeros_like(image)
        for i, slice in enumerate(image):
            mask = np.zeros_like(slice)
            # B = np.argwhere(slice)
            B = torch.nonzero(slice)
            try:
                (ystart, xstart), _ = B.min(0)
                (ystop, xstop), _ = B.max(0)
                (ystop, xstop) = (ystop+1, xstop+1)
                # (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
                mask[ystart:ystop, xstart:xstop] = 1
                mask_3D[i,:,:] = mask
            except:
                mask_3D[i,:,:] = mask
                
        return mask_3D
       
        

    def DRAEM_Aug(self, image, anomaly_source_image):
        aug = self.randAugmenter()
        perlin_scale = 6                        # maybe determine the noise size
        min_perlin_scale = 0
        # anomaly_source_img = self.read_nib_2_cv2(anomaly_source_path)
        # anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(image=anomaly_source_image)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            # objectMask = image > 0.01
            
            ObjectMask = self.getBbox(image)
            NoiseMask = (perlin_thr).astype(np.float32)
            msk = NoiseMask * ObjectMask
            augmented_image = msk * augmented_image + (1-msk)*image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly=0.0
            return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)
    
    def gaussianSeperate(self, x, noise_res=16, noise_std=0.2):
        
        ns = torch.normal(mean=torch.zeros(x.shape[0], noise_res, noise_res), std=noise_std).to(x.device)

        ns = F.upsample_bilinear(ns, size=[x.shape[1], x.shape[2]])

        # Roll to randomly translate the generated noise.
        roll_x = random.choice(range(x.shape[1]))
        roll_y = random.choice(range(x.shape[2]))
        ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])

        # mask = x.sum(dim=1, keepdim=True) > 0.01
        mask = x > 0.01
        ns *= mask # Only apply the noise in the foreground.
        res = x + ns

        return res  
        
    def gaussianUnified(self, x, noise_res=16, noise_std=0.2):
            
        ns = torch.normal(mean=torch.zeros(1, 1, noise_res, noise_res), std=noise_std).to(x.device)

        ns = F.upsample_bilinear(ns, size=[x.shape[1], x.shape[2]])

        # Roll to randomly translate the generated noise.
        roll_x = random.choice(range(x.shape[1]))
        roll_y = random.choice(range(x.shape[2]))
        ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])
        
        
        size_to_slice = random.randint(2, x.shape[0])
        startID = random.randint(0, x.shape[0]-size_to_slice)
        Coef_ascend = np.arange(0,1,1 / ((size_to_slice)//2))
        Coef_descend = np.arange(1,0, -1 / (size_to_slice - size_to_slice//2))
        Coefs = np.zeros(x.shape[0])
        Coefs[startID:startID + size_to_slice//2] = Coef_ascend[:size_to_slice//2]
        Coefs[startID + size_to_slice//2 : size_to_slice + startID] = Coef_descend[:size_to_slice - size_to_slice//2]
        Coefs = Coefs.reshape(-1, 1, 1)           # [256, 1, 1]
        
        # stack the ns for x.shape[0] times
        ns = torch.squeeze(ns, dim=0)
        ns = ns.repeat(x.shape[0], 1, 1)

        # mask = x.sum(dim=1, keepdim=True) > 0.01
        # mask = x > 0.01
        mask = self.getBbox_3D(x)
        ns *= mask # Only apply the noise in the foreground.
        res = x + ns * Coefs

        return res, ns * Coefs, mask
    
    def Circle_Aug(x):
        from skimage import draw
        arr = np.zeros((200, 200))
        rr, cc = draw.circle_perimeter(100, 100, radius=80, shape=arr.shape)
        arr[rr, cc] = 1
        
    def read_nib_2_cv2(self, img_path):
        nimg = nib.load(img_path)
        nimg_array = nimg.get_fdata()           # 3d image
        img_list = []
        for i in range(nimg_array.shape[2]):
            slice = nimg_array[:,:,i]
            slice_arr = (slice*255).astype(np.uint8)
            # slice.arr = np.expand_dims(slice_arr, axis=0)
            img_list.append(slice_arr)
            
        img_arr = np.stack(img_list, axis=0)
        return img_arr
        
        

    def DRAEM_transform(self, image_path, anomaly_source_path):
        # read nibal as cv2 format
        object_3D = self.read_nib_2_cv2(image_path)
        object_3D = cv2.resize(object_3D, dsize=(self.resize_shape[1], self.resize_shape[0]))
        
        anomlaySource_3D = self.read_nib_2_cv2(anomaly_source_path)
        anomlaySource_3D = cv2.resize(anomlaySource_3D, dsize=(self.resize_shape[1], self.resize_shape[0]))

        # do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        # if do_aug_orig:
        #     object_3D = self.rot(object_3D=object_3D)
        
        image_list, augmented_image_list, anomaly_mask_list, has_anomaly_list = [], [], [], []
        for i, image in enumerate(object_3D):
            image = np.expand_dims(image, axis = 2)
            image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
            augmented_image, anomaly_mask, has_anomaly = self.DRAEM_Aug(image, np.expand_dims(anomlaySource_3D[i], axis = 2))
            augmented_image = np.transpose(augmented_image, (2, 0, 1))
            image = np.transpose(image, (2, 0, 1))
            anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
            
            image_list.append(image)
            augmented_image_list.append(augmented_image)
            anomaly_mask_list.append(anomaly_mask)
            has_anomaly_list.append(has_anomaly)
            
        image = np.stack(image_list).squeeze(axis=1)
        augmented_image = np.stack(augmented_image_list, axis=0).squeeze(axis=1)
        anomaly_mask = np.stack(anomaly_mask_list, axis=0).squeeze(axis=1)
        has_anomaly = np.stack(has_anomaly_list, axis=0).squeeze(axis=1)       
        
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        if self.augumentation == 'DRAEM':
            anomaly_source_idx = torch.randint(0, len(self.image_paths), (1,)).item()
            image, augmented_image, anomaly_mask, has_anomaly = self.DRAEM_transform(self.image_paths[idx],
                                                                            self.image_paths[anomaly_source_idx])
            # sample = {'image': image, "anomaly_mask": anomaly_mask,
            #         'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}
            
            return image, anomaly_mask, augmented_image, has_anomaly
            
        
        elif 'gaussian' in self.augumentation:
            
            img_path = self.image_paths[idx]
            nimg = nib.load(img_path)
            nimg_array = nimg.get_fdata()           # 3d image
            img_list = []
            for i in range(nimg_array.shape[2]):
                # slice =  np.expand_dims(nimg_array[:,:,i], axis=0)
                slice = nimg_array[:,:,i]
                # nimg_tensor = self.transform(slice)
                nimg_tensor = torch.tensor(slice)
                nimg_tensor = torch.unsqueeze(nimg_tensor, dim = 0)
                img_list.append(nimg_tensor)
            # img_tensor = torch.tensor(img_list)
            img_tensor = torch.cat(img_list, dim = 0)
            img_tensor = img_tensor.float()             # torch.tensor([256, 256, 256])
            
            aug_tensor, tumor, mask = self.dispatcher[self.augumentation](img_tensor)
            aug_tensor = aug_tensor.float()
            

            return (img_tensor, aug_tensor, mask)
