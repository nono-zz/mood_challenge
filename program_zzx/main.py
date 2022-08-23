# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from tkinter import E
import torch
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from model import Modified3DUNet, DiscriminativeSubNetwork, ReconstructiveSubNetwork
from model_unet3D import UNet3D

from data_loader import TrainDataset, TestDataset

import torch.backends.cudnn as cudnn
import argparse
# from test import evaluation, visualization, test
from torch.nn import functional as F

from tensorboard_visualizer import TensorboardVisualizer

from evaluation import evaluation3D, evaluation2D, evaluationDRAEM
from loss import FocalLoss, SSIM, DiceLoss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def diceLoss(pred, gt):
    intersection = (pred*gt).sum()
    return (2. * intersection)/(pred.sum() + gt.sum())


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train():
    epochs = args.epochs
    learning_rate = args.lr
    batch_size = 1
    image_size = 256
    task = 'brain'
    if task == 'brain':
        channels = 256
    else:
        channels = 512
        
    log_dir = '/home/zhaoxiang/log'
    
    visualizer = TensorboardVisualizer(log_dir=os.path.join(log_dir, args.backbone))
    # ckp_path = os.path.join('/home/zhaoxiang/checkpoints', args.backbone + '_' + args.augumentation + '.pckl')
    ckp_path = os.path.join('/home/zhaoxiang/checkpoints', args.backbone + '.pckl')
        
    n_classes = 2
        
    device = 'cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu'
    print(device)
    data_path = args.data_path
    
    train_data = TrainDataset(root_dir=data_path, size=[256,256], augumentation=args.augumentation)             # what does the ImageFolder stands for?
    test_data = TestDataset(root_dir=data_path, size=[256,256], augumentation=args.augumentation)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)         # learn how torch.utils.data.DataLoader functions
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)


    if args.backbone == '3D':
        # model = Modified3DUNet(1, n_classes, base_n_filter=4)
        model = UNet3D(1, 2, f_maps=8)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5,0.999))
    
    elif args.augumentation != 'DRAEM':

        if args.backbone == '3D':
            # model = Modified3DUNet(1, n_classes, base_n_filter=4)
            model = UNet3D(1, 2, f_maps=8)
        elif args.backbone == '2D':
            model = DiscriminativeSubNetwork(in_channels=1, out_channels=1)
        
        if args.resume_training:
            model.load_state_dict(torch.load(ckp_path)['model'])    
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5,0.999))
        
    
    else:
        model_rec = ReconstructiveSubNetwork(in_channels=1, out_channels=1)
        model_seg = DiscriminativeSubNetwork(in_channels=2, out_channels=2)
   
        if args.resume_training:
            model_rec.load_state_dict(torch.load(ckp_path)['model_rec'])  
            model_seg.load_state_dict(torch.load(ckp_path)['model_seg'])   
            
        model_rec.to(device)
        model_seg.to(device)
        optimizer = torch.optim.Adam(list(model_rec.parameters())+list(model_seg.parameters()), lr=learning_rate, betas=(0.5,0.999))
         

    lossMSE = torch.nn.MSELoss()
    lossCos = torch.nn.CosineSimilarity()
    lossDice = DiceLoss()
    
    
    if args.backbone == '3D':

        for epoch in range(epochs):
            # pixelAP, sampleAP = evaluation3D(args, epoch, device, model, test_dataloader, visualizer)
            model.train()
            loss_list, loss1_list, loss2_list = [], [], []

            # for img, aug, mask in train_dataloader:
            for img, mask, aug, _ in train_dataloader:     
                
                     
            # for img in train_dataloader:                # need to augument the image                          
                img = img.to(device)
                aug = aug.to(device)
                mask = mask.to(device)
                
                x = torch.unsqueeze(aug, dim=1)
                outputs = model(x)
                # samplePred = outputs[0]
                # pixelPred = outputs[1][:,0,:,:,:]
                pixelPred = outputs[:,0,:,:,:]
                maskPred = outputs[:,1,:,:,:]
                
                loss1 = lossMSE(img, pixelPred)
                loss2 = lossDice(maskPred, mask)
                
                loss = loss1+loss2
                # loss = loss1
                # loss = loss2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
                loss1_list.append(loss1.item())
                loss2_list.append(loss2.item())
            print('epoch [{}/{}], loss:{:.6f}'.format(epoch + 1, epochs, np.mean(loss_list)))
            print('Reconstruction loss:{}       Segmentation loss:{}'.format(np.mean(loss1_list), np.mean(loss2_list)))
            
            
            # visualization
            visualizer.visualize_image_batch(img[0,50], epoch, image_name='img_50')
            visualizer.visualize_image_batch(img[0,125], epoch, image_name='img_125')
            visualizer.visualize_image_batch(img[0,200], epoch, image_name='img_200')
            visualizer.visualize_image_batch(aug[0,50], epoch, image_name='aug_50')
            visualizer.visualize_image_batch(aug[0,125], epoch, image_name='aug_125')
            visualizer.visualize_image_batch(aug[0,200], epoch, image_name='aug_200')    
            visualizer.visualize_image_batch(pixelPred[0,50], epoch, image_name='out_50')
            visualizer.visualize_image_batch(pixelPred[0,125], epoch, image_name='out_125')
            visualizer.visualize_image_batch(pixelPred[0,200], epoch, image_name='out_200')    
            visualizer.visualize_image_batch(maskPred[0,125], epoch, image_name='seg_125')
            visualizer.visualize_image_batch(mask[0,50], epoch, image_name='mask_50')
            visualizer.visualize_image_batch(mask[0,125], epoch, image_name='mask_125')
            visualizer.visualize_image_batch(mask[0,200], epoch, image_name='mask_200')  
            
                
            # after training evaluation
            # model.eval()
            with torch.no_grad():
                for img, mask, aug, _ in train_dataloader:     
                    
                        
                # for img in train_dataloader:                # need to augument the image                          
                    img = img.to(device)
                    aug = aug.to(device)
                    mask = mask.to(device)
                    
                    x = torch.unsqueeze(aug, dim=1)
                    outputs = model(x)
                    # samplePred = outputs[0]
                    # pixelPred = outputs[1][:,0,:,:,:]
                    pixelPred = outputs[:,0,:,:,:]
                    maskPred = outputs[:,1,:,:,:]
                    
                    loss1 = lossMSE(img, pixelPred)
                    loss2 = lossDice(maskPred, mask)
                    
                    loss = loss1+loss2
                    # loss = loss1
                    # loss = loss2
                    loss_list.append(loss.item())
                    loss1_list.append(loss1.item())
                    loss2_list.append(loss2.item())
                print('epoch [{}/{}], error:{:.6f}'.format(epoch + 1, epochs, np.mean(loss_list)))
                print('Reconstruction error:{}       Segmentation error:{}'.format(np.mean(loss1_list), np.mean(loss2_list)))
                
                
                # visualization
                visualizer.visualize_image_batch(img[0,50], epoch, image_name='img_50')
                visualizer.visualize_image_batch(img[0,125], epoch, image_name='img_125')
                visualizer.visualize_image_batch(img[0,200], epoch, image_name='img_200')
                visualizer.visualize_image_batch(aug[0,50], epoch, image_name='aug_50')
                visualizer.visualize_image_batch(aug[0,125], epoch, image_name='aug_125')
                visualizer.visualize_image_batch(aug[0,200], epoch, image_name='aug_200')    
                visualizer.visualize_image_batch(pixelPred[0,50], epoch, image_name='out_50')
                visualizer.visualize_image_batch(pixelPred[0,125], epoch, image_name='out_125')
                visualizer.visualize_image_batch(pixelPred[0,200], epoch, image_name='out_200')    
                visualizer.visualize_image_batch(maskPred[0,125], epoch, image_name='seg_125')
                visualizer.visualize_image_batch(mask[0,50], epoch, image_name='mask_50')
                visualizer.visualize_image_batch(mask[0,125], epoch, image_name='mask_125')
                visualizer.visualize_image_batch(mask[0,200], epoch, image_name='mask_200')  
                
                
                if (epoch) % 5 == 0:
                    pixelAP, sampleAP = evaluation3D(args, epoch, device, model, test_dataloader, visualizer)
                print('Pixel Average Precision:{:.4f}, Sample Average Precision:{:.4f}'.format(pixelAP, sampleAP))
                torch.save({'model': model.state_dict()}, ckp_path)
                
            
    elif args.backbone == '2D':
        if args.augumentation == 'DRAEM':
            loss_l2 = torch.nn.modules.loss.MSELoss()
            loss_ssim = SSIM(device=device)
            loss_focal = FocalLoss()
            loss_list = []
            
            
            for epoch in range(epochs):
                model_rec.train()
                model_seg.train()
                for images, anomaly_masks, augmented_images, has_anomalies in train_dataloader:
                    for i in range(images.shape[2]):
                        image = images[:,i,:,:].unsqueeze(dim = 1)
                        anomaly_mask = anomaly_masks[:,i,:,:].unsqueeze(dim = 1)
                        augmented_image = augmented_images[:,i,:,:].unsqueeze(dim = 1)
                        has_anomaly = has_anomalies[:,i]
                        
                        image = image.to(device)
                        augmented_image = augmented_image.to(device)
                        anomaly_mask = anomaly_mask.to(device)

                        image_rec = model_rec(augmented_image)
                        joined_in = torch.cat((image_rec, augmented_image), dim=1)

                        out_mask = model_seg(joined_in)
                        out_mask_sm = torch.softmax(out_mask, dim=1)

                        l2_loss = loss_l2(image_rec,image)
                        ssim_loss = loss_ssim(image_rec, image)

                        segment_loss = loss_focal(out_mask_sm, anomaly_mask)
                        loss = l2_loss + ssim_loss + segment_loss

                        optimizer.zero_grad()

                        loss.backward()
                        optimizer.step()
                        
                        
                        loss_list.append(loss.item())
                        
                        # save_image(image, 'image.png')
                        # save_image(augmented_image, 'augmented_image.png')
                        # save_image(image_rec, 'image_rec.png')
                        # save_image(out_mask_sm[:, 1:, :, :], 'out_mask_sm.png')
                        # save_image(anomaly_mask, 'anomlay_mask.png')
                    
               
                        if i in [50, 125, 200]:
                            # visualize
                            visualizer.visualize_image_batch(images[0,i], epoch, image_name='DRAEM_raw_{}'.format(i))
                            # visualizer.visualize_image_batch(images[0,125], epoch, image_name='img_125')
                            # visualizer.visualize_image_batch(images[0,200], epoch, image_name='img_200')
                            visualizer.visualize_image_batch(augmented_images[0,i], epoch, image_name='DRAEM_aug_{}'.format(i))
                            # visualizer.visualize_image_batch(augmented_images[0,125], epoch, image_name='aug_125')
                            # visualizer.visualize_image_batch(augmented_images[0,200], epoch, image_name='aug_200')    
                            visualizer.visualize_image_batch(image_rec, epoch, image_name='DRAEM_rec_{}'.format(i))
                            visualizer.visualize_image_batch(out_mask_sm, epoch, image_name='DRAEM_seg_{}'.format(i))
                            visualizer.visualize_image_batch(anomaly_mask, epoch, image_name='DRAEM_mask_{}'.format(i))
                            
                            # visualizer.visualize_image_batch(outputs[0,125], epoch, image_name='out_125')
                            # visualizer.visualize_image_batch(outputs[0,200], epoch, image_name='out_200')  
                    
                print('epoch [{}/{}], loss:{:.6f}'.format(epoch + 1, epochs, np.mean(loss_list)))
                
                
                model_rec.train()
                model_seg.train()
                for images, anomaly_masks, augmented_images, has_anomalies in train_dataloader:
                    for i in range(images.shape[2]):
                        image = images[:,i,:,:].unsqueeze(dim = 1)
                        anomaly_mask = anomaly_masks[:,i,:,:].unsqueeze(dim = 1)
                        augmented_image = augmented_images[:,i,:,:].unsqueeze(dim = 1)
                        has_anomaly = has_anomalies[:,i]
                        
                        image = image.to(device)
                        augmented_image = augmented_image.to(device)
                        anomaly_mask = anomaly_mask.to(device)

                        image_rec = model_rec(augmented_image)
                        joined_in = torch.cat((image_rec, augmented_image), dim=1)

                        out_mask = model_seg(joined_in)
                        out_mask_sm = torch.softmax(out_mask, dim=1)

                        l2_loss = loss_l2(image_rec,image)
                        ssim_loss = loss_ssim(image_rec, image)

                        segment_loss = loss_focal(out_mask_sm, anomaly_mask)
                        loss = l2_loss + ssim_loss + segment_loss


                        
                        
                        loss_list.append(loss.item())
                        
                        save_image(image, 'image.png')
                        save_image(augmented_image, 'augmented_image.png')
                        save_image(image_rec, 'image_rec.png')
                        save_image(out_mask_sm[:, 1:, :, :], 'out_mask_sm.png')
                        save_image(anomaly_mask, 'anomlay_mask.png')
                    
               
                        # if i in [50, 125, 200]:
                        #     # visualize
                        #     visualizer.visualize_image_batch(images[0,i], epoch, image_name='DRAEM_raw_{}'.format(i))
                        #     # visualizer.visualize_image_batch(images[0,125], epoch, image_name='img_125')
                        #     # visualizer.visualize_image_batch(images[0,200], epoch, image_name='img_200')
                        #     visualizer.visualize_image_batch(augmented_images[0,i], epoch, image_name='DRAEM_aug_{}'.format(i))
                        #     # visualizer.visualize_image_batch(augmented_images[0,125], epoch, image_name='aug_125')
                        #     # visualizer.visualize_image_batch(augmented_images[0,200], epoch, image_name='aug_200')    
                        #     visualizer.visualize_image_batch(image_rec, epoch, image_name='DRAEM_rec_{}'.format(i))
                        #     visualizer.visualize_image_batch(out_mask_sm, epoch, image_name='DRAEM_seg_{}'.format(i))
                        #     visualizer.visualize_image_batch(anomaly_mask, epoch, image_name='DRAEM_mask_{}'.format(i))
                            
                        #     # visualizer.visualize_image_batch(outputs[0,125], epoch, image_name='out_125')
                        #     # visualizer.visualize_image_batch(outputs[0,200], epoch, image_name='out_200')  
                    
                print('epoch [{}/{}], error:{:.6f}'.format(epoch + 1, epochs, np.mean(loss_list)))
                
                
                pixelAP, sampleAP = evaluationDRAEM(args, epoch, device, model_rec, model_seg, test_dataloader, visualizer)
                torch.save({'model_rec': model_rec.state_dict(),
                    'model_seg': model_seg.state_dict()}, ckp_path)
        
        else:
        
            for epoch in range(epochs):
                loss_list = []
                # pixelAP, sampleAP = evaluation2D(args, epoch, device, model, test_dataloader, visualizer)
                model.train()
                for img, aug, mask in train_dataloader:
                    img = img.to(device)
                    aug = aug.to(device)
                    outputs = torch.zeros_like(img)  
                    
                    for i in range(img.shape[2]):
                        raw = img[:,i,:,:]
                        raw = torch.unsqueeze(raw, dim=1)
                        aug_slice = aug[:,i,:,:]
                        aug_slice = torch.unsqueeze(aug_slice, dim=1)

                        output_slice = model(aug_slice)
                        # output_slice = torch.squeeze(output_slice, dim=1)
                        outputs[:,i,:,:] = output_slice
                        
                        loss1 =  lossMSE(raw, output_slice)
                        loss2 = torch.mean(1- lossCos(raw,output_slice))
                        loss = loss1+loss2
                        loss = loss1
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        loss_list.append(loss.item())
                        
                        
                print('epoch [{}/{}], loss:{:.6f}'.format(epoch + 1, epochs, np.mean(loss_list)))
            
                # visualization
                visualizer.visualize_image_batch(img[0,50], epoch, image_name='img_50')
                visualizer.visualize_image_batch(img[0,125], epoch, image_name='img_125')
                visualizer.visualize_image_batch(img[0,200], epoch, image_name='img_200')
                visualizer.visualize_image_batch(aug[0,50], epoch, image_name='aug_50')
                visualizer.visualize_image_batch(aug[0,125], epoch, image_name='aug_125')
                visualizer.visualize_image_batch(aug[0,200], epoch, image_name='aug_200')    
                visualizer.visualize_image_batch(outputs[0,50], epoch, image_name='out_50')
                visualizer.visualize_image_batch(outputs[0,125], epoch, image_name='out_125')
                visualizer.visualize_image_batch(outputs[0,200], epoch, image_name='out_200')    
                
                
                print('Pixel Average Precision:{:.4f}, Sample Average Precision:{:.4f}'.format(pixelAP, sampleAP))
                torch.save({'model': model.state_dict()}, ckp_path)
            


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default = 0.001, action='store', type=int)
    parser.add_argument('--epochs', default=80, action='store', type=int)
    parser.add_argument('--data_path', default='/home/zhaoxiang/mood_challenge_data/data', type=str)
    parser.add_argument('--checkpoint_path', default='/checkpoints/', action='store', type=str)
    parser.add_argument('--img_size', default=256, action='store')
    
    
    parser.add_argument('--loss_mode', default='MSE', action='store', choices = ['MSE', 'Cos', 'MSE_Cos'])
    parser.add_argument('--gpu_id', default=0, action='store', type=int, required=False)
    parser.add_argument('--augumentation', default='gaussianUnified', action='store',choices = ['gaussianSeperate', 'gaussianUnified', 'Circle', 'DRAEM'])
    parser.add_argument('--task', default='Brain', action='store',choices = ['Brain', 'Abdom'])
    parser.add_argument('--backbone', default='2D', action='store',choices = ['3D', '2D'])
    parser.add_argument('--resume_training', default=False, type = bool)
    
    
    args = parser.parse_args()


    setup_seed(111)
    train()

  