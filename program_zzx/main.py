# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import torch
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from model import Modified3DUNet, DiscriminativeSubNetwork
from model_unet3D import UNet3D

from data_loader import TrainDataset, TestDataset

import torch.backends.cudnn as cudnn
import argparse
# from test import evaluation, visualization, test
from torch.nn import functional as F

from tensorboard_visualizer import TensorboardVisualizer

from evaluation import evaluation3D, evaluation2D

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
        model = UNet3D(1, 1, f_maps=8)
    elif args.backbone == '2D':
        model = DiscriminativeSubNetwork(in_channels=1, out_channels=1)
        
    if args.resume_training:
        model.load_state_dict(torch.load(ckp_path)['model'])    
    model.to(device)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5,0.999))
    lossMSE = torch.nn.MSELoss()
    lossCos = torch.nn.CosineSimilarity()
    
    
    if args.backbone == '3D':

        for epoch in range(epochs):
            pixelAP, sampleAP = evaluation3D(args, epoch, device, model, test_dataloader, visualizer)
            model.train()
            loss_list = []
            for img, aug in train_dataloader:         # where does the label come from? torch.Size([16]) why is it 16?
            # for img in train_dataloader:                # need to augument the image                          
                img = img.to(device)
                aug = aug.to(device)
                
                x = torch.unsqueeze(img, dim=1)
                outputs = model(x)
                # samplePred = outputs[0]
                # pixelPred = outputs[1][:,0,:,:,:]
                pixelPred = outputs[:,0,:,:,:]
                
                loss1 = lossMSE(img, pixelPred)
                # loss2 = torch.mean(1 - lossCos(img,pixelPred))
                # loss2 = torch.mean(1 - lossCos(img,pixelPred))
                
                # loss = loss1+loss2
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
            visualizer.visualize_image_batch(pixelPred[0,50], epoch, image_name='out_50')
            visualizer.visualize_image_batch(pixelPred[0,125], epoch, image_name='out_125')
            visualizer.visualize_image_batch(pixelPred[0,200], epoch, image_name='out_200')    
            
            if (epoch + 1) % 10 == 0:
                pixelAP, sampleAP = evaluation3D(args, epoch, device, model, test_dataloader, visualizer)
                print('Pixel Average Precision:{:.4f}, Sample Average Precision:{:.4f}'.format(pixelAP, sampleAP))
                torch.save({'model': model.state_dict()}, ckp_path)

            
    elif args.backbone == '2D':
        
        for epoch in range(epochs):
            loss_list = []
            pixelAP, sampleAP = evaluation2D(args, epoch, device, model, test_dataloader, visualizer)
            model.train()
            for img, aug in train_dataloader:
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
            
            
            
                
                
                
                
                # # for i in range(img.shape[2]):
                # for i in range(10):
                #     raw = img[:,i,:,:]
                #     aug_slice = aug[:,i,:,:]
                #     aug_slice = torch.unsqueeze(aug_slice, dim=1)

                #     output_slice = model(aug_slice)
                #     output_slice = torch.squeeze(output_slice, dim=1)
                #     outputs[:,i,:,:] = output_slice
                    
                    
                    
                
                # loss1 =  lossMSE(img, outputs)
                # loss2 = torch.mean(1- lossCos(img,outputs))
                # loss = loss1+loss2
                # loss = loss1
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                # loss_list.append(loss.item())
        
        
     



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default = 0.001, action='store', type=int)
    parser.add_argument('--epochs', default=80, action='store', type=int)
    parser.add_argument('--data_path', default='/home/zhaoxiang/mood_challenge_data/data', type=str)
    parser.add_argument('--checkpoint_path', default='/checkpoints/', action='store', type=str)
    parser.add_argument('--img_size', default=256, action='store')
    
    
    parser.add_argument('--loss_mode', default='MSE', action='store', choices = ['MSE', 'Cos', 'MSE_Cos'])
    parser.add_argument('--gpu_id', default=1, action='store', type=int, required=False)
    parser.add_argument('--augumentation', default='DRAEM', action='store',choices = ['gaussianSeperate', 'gaussianUnified', 'Circle', 'DRAEM'])
    parser.add_argument('--task', default='Brain', action='store',choices = ['Brain', 'Abdom'])
    parser.add_argument('--backbone', default='2D', action='store',choices = ['3D', '2D'])
    parser.add_argument('--resume_training', default=False, type = bool)
    
    
    args = parser.parse_args()


    setup_seed(111)
    train()

  