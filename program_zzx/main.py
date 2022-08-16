# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import torch
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from model import Modified3DUNet

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

def train(_class_):
    print(_class_)
    epochs = 200
    learning_rate = 0.005
    batch_size = 16
    image_size = 256
        
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(device)
    main_path = '/home/zhaoxiang/mood_challenge_data/data'
    
    train_data = TrainDataset(root_dir=main_path, size=256)             # what does the ImageFolder stands for?
    # test_data = TestDataset(root=main_path, size=256, gt_transform=gt_transform, phase="test")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)         # learn how torch.utils.data.DataLoader functions
    # test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    model = model()
    model.to(device)

    optimizer = torch.optim.Adam(list(model), lr=learning_rate, betas=(0.5,0.999))


    for epoch in range(epochs):
        model.train()
        loss_list = []
        for aug, img in train_dataloader:         # where does the label come from? torch.Size([16]) why is it 16?
            img = img.to(device)
            outputs = model(aug)
            
            loss = (img, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
        # if (epoch + 1) % 10 == 0:
        #     auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device)
        #     ap = evaluation(model, train_dataloader, device)
        #     print('Average Precision:{:.3f}'.format(ap))
        #     torch.save({'model': model.state_dict()}, ckp_path)




if __name__ == '__main__':

    setup_seed(111)
    item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                 'transistor', 'metal_nut', 'screw','toothbrush', 'zipper', 'tile', 'wood']
    for i in item_list:
        train(i)

