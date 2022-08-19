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

from data_loader import TrainDataset, TestDataset

import torch.backends.cudnn as cudnn
import argparse
# from test import evaluation, visualization, test
from torch.nn import functional as F

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
        
    n_classes = 2
        
    device = 'cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu'
    print(device)
    data_path = args.data_path
    
    train_data = TrainDataset(root_dir=data_path, size=256, augumentation=args.augumentation)             # what does the ImageFolder stands for?
    # test_data = TestDataset(root=main_path, size=256, gt_transform=gt_transform, phase="test")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)         # learn how torch.utils.data.DataLoader functions
    # test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    # model = Modified3DUNet(channels, n_classes)
    model = DiscriminativeSubNetwork(in_channels=channels, out_channels=channels, base_channels=4)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5,0.999))
    lossFN = torch.nn.MSELoss()


    for epoch in range(epochs):
        model.train()
        loss_list = []
        for img, aug in train_dataloader:         # where does the label come from? torch.Size([16]) why is it 16?
        # for img in train_dataloader:                # need to augument the image                          
            img = img.to(device)
            aug = aug.to(device)
            outputs = model(aug)
            # outputs = model(img)
            
            loss =  lossFN(img, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
        
        
        # visualization
        
        
        
        
        
        
        # if (epoch + 1) % 10 == 0:
        #     auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device)
        #     ap = evaluation(model, train_dataloader, device)
        #     print('Average Precision:{:.3f}'.format(ap))
        #     torch.save({'model': model.state_dict()}, ckp_path)




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default = 0.001, action='store', type=int)
    parser.add_argument('--epochs', default=80, action='store', type=int)
    parser.add_argument('--data_path', default='/home/zhaoxiang/mood_challenge_data/data', type=str)
    parser.add_argument('--checkpoint_path', default='/checkpoints/', action='store', type=str)
    parser.add_argument('--backbone', default='crossPred', action='store',choices = ['crossPred', 'DRAEM'])
    parser.add_argument('--img_size', default=256, action='store')
    
    
    parser.add_argument('--loss_mode', default='MSE', action='store', choices = ['MSE', 'Cos', 'MSE_Cos'])
    parser.add_argument('--gpu_id', default=0, action='store', type=int, required=False)
    parser.add_argument('--augumentation', default='gaussianUnified', action='store',choices = ['gaussianSeperate', 'gaussianUnified', 'Circle'])
    parser.add_argument('--task', default='Brain', action='store',choices = ['Brain', 'Abdom'])
    
    
    args = parser.parse_args()


    setup_seed(111)
    train()

