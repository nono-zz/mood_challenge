from matplotlib.pyplot import gray
import torch
from torch.utils.data import DataLoader
from torch import optim
from tensorboard_visualizer import TensorboardVisualizer
from loss import FocalLoss, SSIM
import os
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
import numpy as np

import torch.nn.functional as F
import random

from dataloader_zzx import MVTecDataset, Medical_dataset
from evaluation_mood import evaluation
from cutpaste import CutPaste3Way, CutPasteUnion


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def mean(list_x):
    return sum(list_x)/len(list_x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
def get_data_transforms(size, isize):
    # mean_train = [0.485]         # how do you set the mean_train and std_train in the get_data_transforms function?
    # mean_train = [-0.1]
    # std_train = [0.229]
    data_transforms = transforms.Compose([
        # transforms.Resize((size, size)),
        # transforms.CenterCrop(isize),
        
        #transforms.CenterCrop(args.input_size),
        transforms.ToTensor()
        # transforms.Normalize(mean=mean_train,
        #                      std=std_train)
    ])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])

    return data_transforms, gt_transforms

        
        
def add_Gaussian_noise(x, noise_res, noise_std, img_size):
    ns = torch.normal(mean=torch.zeros(x.shape[0], x.shape[1], noise_res, noise_res), std=noise_std).to(x.device)

    ns = F.upsample_bilinear(ns, size=[img_size, img_size])

    # Roll to randomly translate the generated noise.
    roll_x = random.choice(range(128))
    roll_y = random.choice(range(128))
    ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])

    mask = x.sum(dim=1, keepdim=True) > 0.01
    ns *= mask # Only apply the noise in the foreground.
    res = x + ns
    
    return res
        

def train_on_device(args):

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # run_name = args.experiment_name + '_' +str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_"+"Guassian_blur"
    run_name = args.experiment_name + '_' +str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_" + args.model + "_" + args.process_method

    visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))
    main_path = '/home/zhaoxiang/dataset/{}'.format(args.dataset_name)
    
    data_transform, gt_transform = get_data_transforms(args.img_size, args.img_size)
    test_transform, _ = get_data_transforms(args.img_size, args.img_size)
    train_transform = transforms.Compose([])
    train_transform.transforms.append(CutPaste3Way(transform = test_transform))
    # test_transform, _ = get_data_transforms(args.img_size, args.img_size)

    dirs = os.listdir(main_path)
    
    for dir_name in dirs:
        if 'train' in dir_name:
            train_dir = dir_name
        elif 'test' in dir_name:
            if 'label' in dir_name:
                label_dir = dir_name
            else:
                test_dir = dir_name
    if 'label_dir' in locals():
        dirs = [train_dir, test_dir, label_dir]                


    from model_noise import UNet
    
    # device = torch.device('cuda:{}'.format(args.gpu_id))
    device = None
    n_input = 1
    n_classes = 1           # the target is the reconstructed image
    depth = 4
    wf = 6
    
    if args.model == 'ws_skip_connection':
        model = UNet(in_channels=n_input, n_classes=n_classes, norm="group", up_mode="upconv", depth=depth, wf=wf, padding=True).cuda()
    elif args.model == 'DRAEM_reconstruction':
        model = ReconstructiveSubNetwork(in_channels=n_input, out_channels=n_input).cuda()
    elif args.model == 'DRAEM_discriminitive':
        model = DiscriminativeSubNetwork(in_channels=n_input, out_channels=n_input).cuda()
        
    if args.resume_training:
        base_path= '/home/zhaoxiang/baselines/pretrain'
        output_path = os.path.join(base_path, 'output')

        experiment_path = os.path.join(output_path, run_name)
        ckp_path = os.path.join(experiment_path, 'last.pth')
        
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    
    
    train_data = MVTecDataset(root=main_path, transform = test_transform, gt_transform=gt_transform, phase='train', dirs = dirs, data_source=args.experiment_name, args = args)
    val_data = MVTecDataset(root=main_path, transform = test_transform, gt_transform=gt_transform, phase='test', dirs = dirs, data_source=args.experiment_name)
    test_data = MVTecDataset(root=main_path, transform = test_transform, gt_transform=gt_transform, phase='test', dirs = dirs, data_source=args.experiment_name)
        
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = args.bs, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size = args.bs, shuffle = False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = 1, shuffle = False)
        
    loss_l1 = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        # evaluation(args, model, test_dataloader, epoch, loss_l1, visualizer, run_name)
        model.train()
        loss_list = []
        # for img, label, img_path in train_dataloader:         
        for img, aug in train_dataloader:
            img = img.squeeze(dim = 0)
            aug = aug.squeeze(dim = 0)
            
            img = img.cuda()
            aug = aug.cuda()
            
            # if "Gaussian" in args.process_method:
            #     input = add_Gaussian_noise(img, args.noise_res, args.noise_std, args.img_size)         # if noise -> reconstruction

            output = model(aug)
            
            save_image(aug, 'input.png')
            save_image(output, 'output.png')
            save_image(img, 'target.png')
            loss = loss_l1(img, output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            loss_list.append(loss.item())
            
        print('epoch [{}/{}], loss:{:.4f}'.format(args.epochs, epoch, mean(loss_list)))
        
        
        visualizer.plot_loss(mean(loss_list), epoch, loss_name='L1_loss')
        visualizer.visualize_image_batch(aug, epoch, image_name='input')
        visualizer.visualize_image_batch(img, epoch, image_name='target')
        visualizer.visualize_image_batch(output, epoch, image_name='output')
        
        if (epoch) % 3 == 0:
            model.eval()
            error_list = []
            for img, gt, label, img_path, saves in val_dataloader:
                img = img.cuda()
                input = img
                output = model(input)
                loss = loss_l1(input, output)
                
                save_image(input, 'input_eval.png')
                save_image(output, 'output_eval.png')
                save_image(img, 'target_eval.png')
                
                error_list.append(loss.item())
            
            print('eval [{}/{}], error:{:.4f}'.format(args.epochs, epoch, mean(error_list)))
            visualizer.plot_loss(mean(error_list), epoch, loss_name='L1_loss_eval')
            visualizer.visualize_image_batch(input, epoch, image_name='target_eval')
            visualizer.visualize_image_batch(output, epoch, image_name='output_eval')
            
        if (epoch) % 10 == 0:
            model.eval()
            evaluation(args, model, test_dataloader, epoch, loss_l1, visualizer, run_name)
                
        
        

if __name__=="__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', default=1,  action='store', type=int)
    parser.add_argument('--lr', default=0.0001, action='store', type=float)
    parser.add_argument('--epochs', default=700, action='store', type=int)
    parser.add_argument('--c v', default='/home/zhaoxiang/baselines/DRAEM/datasets/mvtec/', action='store', type=str)
    parser.add_argument('--anomaly_source_path', default='/home/zhaoxiang/baselines/DRAEM/datasets/dtd/images/', action='store', type=str)
    parser.add_argument('--checkpoint_path', default='./checkpoints/', action='store', type=str)
    parser.add_argument('--log_path', default='./logs/', action='store', type=str)
    parser.add_argument('--visualize', default=True, action='store_true')

    parser.add_argument('--backbone', default='noise', action='store')
    
    # for noise autoencoder
    parser.add_argument("-nr", "--noise_res", type=float, default=16,  help="noise resolution.")
    parser.add_argument("-ns", "--noise_std", type=float, default=0.2, help="noise magnitude.")
    parser.add_argument("-img_size", "--img_size", type=float, default=256, help="noise magnitude.")
    
    # need to be changed/checked every time
    parser.add_argument('--bs', default = 1, action='store', type=int)
    parser.add_argument('--gpu_id', default = ['0','1'], action='store', type=str, required=False)
    parser.add_argument('--experiment_name', default='mood_cv2', choices=['retina, liver, brain, head', 'chest'], action='store')
    parser.add_argument('--dataset_name', default='Mood_brain_cv2', choices=['hist_DIY', 'Brain_MRI', 'Head_CT', 'CovidX', 'RESC_average'], action='store')
    parser.add_argument('--model', default='ws_skip_connection', choices=['ws_skip_connection', 'DRAEM_reconstruction', 'DRAEM_discriminitive'], action='store')
    parser.add_argument('--process_method', default='Multi_randomShape_wo_distortion', choices=['none', 'Gaussian_noise', 'DRAEM_natural', 'DRAEM_tumor', 'Simplex_noise', 'Simplex_noise_best_best'], action='store')
    parser.add_argument('--resume_training', default = True, action='store', type=int)
    
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    if args.gpu_id is None:
        gpus = "0"
        os.environ["CUDA_VISIBLE_DEVICES"]= gpus
    else:
        gpus = ""
        for i in range(len(args.gpu_id)):
            gpus = gpus + args.gpu_id[i] + ","
        os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

    # with torch.cuda.device(args.gpu_id):
    train_on_device(args)

