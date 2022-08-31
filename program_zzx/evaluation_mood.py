import torch
import os
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

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



def mean(list_x):
    return sum(list_x)/len(list_x)

def cal_distance_map(input, target):
    # input = np.squeeze(input, axis=0)
    # target = np.squeeze(target, axis=0)
    d_map = np.full_like(input, 0)
    d_map = np.square(input - target)
    return d_map

def dice(pred, gt):
    intersection = (pred*gt).sum()
    return (2. * intersection)/(pred.sum() + gt.sum())

def evaluation(args, model, test_dataloader, epoch, loss_l1, visualizer, run_name, device=None):
    
    base_path= '/home/zhaoxiang/baselines/pretrain'
    output_path = os.path.join(base_path, 'output')
 
    experiment_path = os.path.join(output_path, run_name)
    ckp_path = os.path.join(experiment_path, 'last.pth')
    result_path = os.path.join(experiment_path, 'results.txt')

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)     
    
    
    if args.experiment_name == 'retina':
        names = names = ['564', '572', '584', '590', '597', '640', '682']
    elif args.experiment_name == 'liver':       # liver
        names = ['liver_1_59', 'liver_1_60', 'liver_1_66', 'liver_1_67', 'liver_2_415', 'liver_2_452', 'liver_3_394', 'liver_4_566', 'liver_4_457', 'liver_6_396', 'liver_8_448', 'liver_7_487', 'liver_10_328', 
                            'liver_10_335' 'liver_10_356', 'liver_10_379',  'liver_10_295', 'liver_10_422' , 'liver_16_374', 'liver_16_409', 'liver_18_413', 'liver_19_440', 'liver_20_487', 'liver_21_388', 'liver_23_335',
                            'liver_26_317', 'liver_27_414', 'liver_27_559', 'liver_13_393', 'liver_12_412', 'liver_11_413']
        
    elif args.dataset_name == 'Brain_MRI':
        names = ['Y1.jpg','Y2.jpg','Y10.jpg','Y30.jpg','Y100.jpg',
                    '1 no', '2 no', '10 no', 'N21']
                
    elif args.dataset_name == 'Head_CT':
        names = ['N180.png','N185.png','N189.png'
                'Y040', 'Y051', 'Y070', 'Y090']
        
    elif args.dataset_name == 'CovidX':
        names = ['14d81f378173b86cc53f21d2d67040_jumbo', 'A027284-12-31-1900-NA-CHEST_AP_PORT-38375-4.000000-AP-32977-1-1', 'A619446-01-05-1901-NA-CHEST_AP_PORT-15991-2.000000-AP-94836-1-1',
                     '2ef4cfe7-bbc1-4ce2-ba1e-6df54a94252c', '6c2cf7d6-4e1d-4846-9658-089f74d2cfbc', 'a0af1a60-0f1c-4d53-9a71-dd438aa06b70']
    elif 'Mood' in args.dataset_name:
        names = ['1_69', '2_200', '3_200','0_125', '0_0']
    model.eval()
    error_list = []
    pixel_pred_list, gt_list, sample_pred_list, label_list = [], [], [], []
    for img, gt, label, img_path, saves in test_dataloader:
    # for img, label, img_path in test_dataloader:
        img = img.cuda()
        input = img
        output = model(input)
        loss = loss_l1(input, output)
        
        error_list.append(loss.item())
        
        # illustrate the images first
        # reconstruction, difference, raw, gt
        
        difference = cal_distance_map(output[0,0,:,:].to('cpu').detach().numpy(), input[0,0,:,:].to('cpu').detach().numpy())
        
        pixel_pred_list.extend(difference.ravel())
        gt_list.extend(gt.to('cpu').detach().numpy().ravel())
        # sample
        sample_pred_list.append(np.sum(difference))
        label_list.append(label.item())
        
        # for mood only
        difference = cal_distance_map(output[0,0,:,:].to('cpu').detach().numpy(), input[0,0,:,:].to('cpu').detach().numpy())
        cv2.imwrite('/home/zhaoxiang/baselines/difference.png', difference * 255)

        raw = input[0,0,:,:].to('cpu').detach().numpy()  
        cv2.imwrite('/home/zhaoxiang/baselines/raw.png', raw * 255)
        
        reconstruction = output[0,0,:,:].to('cpu').detach().numpy()
        cv2.imwrite('/home/zhaoxiang/baselines/reconstruction.png', reconstruction * 255)
        
        gt_img = gt[0,0,:,:].to('cpu').detach().numpy()  
        cv2.imwrite('/home/zhaoxiang/baselines/gt.png', gt_img * 255)
                
        for name in names:
            if name in img_path[0]:
                folder_path = os.path.join(experiment_path, name)
                reconstruction_path = os.path.join(folder_path, 'reconstruction_{}.png'.format(epoch))
                difference_path = os.path.join(folder_path, 'difference_{}.png'.format(epoch))
                raw_path = os.path.join(folder_path, 'raw.png')
                gt_path = os.path.join(folder_path, 'gt.png')
                
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)     
                
                reconstruction = output[0,0,:,:].to('cpu').detach().numpy()
                cv2.imwrite(reconstruction_path, reconstruction * 255)
                
                # difference = (output - input)[0,0,:,:].to('cpu').detach().numpy()
                difference = cal_distance_map(output[0,0,:,:].to('cpu').detach().numpy(), input[0,0,:,:].to('cpu').detach().numpy())
                cv2.imwrite(difference_path, difference * 255)
                
                raw = input[0,0,:,:].to('cpu').detach().numpy()  
                cv2.imwrite(raw_path, raw * 255)
                
                if not args.dataset_name in ['Brain_MRI', 'Head_CT', 'CovidX']:
                    gt = gt[0,0,:,:].to('cpu').detach().numpy()  
                    cv2.imwrite(gt_path, gt * 255)
        
    
    print('eval [{}/{}], loss:{:.4f}'.format(args.epochs, epoch, mean(error_list)))
    with open(result_path, 'a') as f:
        f.writelines('eval [{}/{}], loss:{:.4f} \n'.format(args.epochs, epoch, mean(error_list)))
    
    
    torch.save(model.state_dict(), ckp_path)
    
    
    pixelAP = average_precision_score(gt_list, pixel_pred_list)
    sampleAP = average_precision_score(label_list, sample_pred_list)
    print('Pixel Average Precision:{:.4f}, Sample Average Precision:{:.4f}'.format(pixelAP, sampleAP))
    with open(result_path, 'a') as f:
        f.writelines('Pixel Average Precision:{:.4f}, Sample Average Precision:{:.4f}'.format(pixelAP, sampleAP))
        
        
        
        
if __name__ == '__main__':
    import argparse
    from model_noise import UNet
    
    from dataloader_zzx import MVTecDataset
    from torch.utils.data import DataLoader
    from tensorboard_visualizer import TensorboardVisualizer
    
    

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
    
    parser.add_argument('--augmentation', default = 'cp', action='store', type=int)
    
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
    
    data_transform, gt_transform = get_data_transforms(args.img_size, args.img_size)
    main_path = '/home/zhaoxiang/dataset/{}'.format(args.dataset_name)
    
    dirs = os.listdir(main_path)
    
    for dir_name in dirs:
        if 'train' in dir_name:
            train_dir = dir_name
        elif 'test_{}'.format(args.augmentation) in dir_name:
            if 'label' in dir_name:
                label_dir = dir_name
            else:
                test_dir = dir_name
    if 'label_dir' in locals():
        dirs = [train_dir, test_dir, label_dir]                

    
    
    device = None
    n_input = 1
    n_classes = 1           # the target is the reconstructed image
    depth = 4
    wf = 6
    
    loss_l1 = torch.nn.L1Loss()
    visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))
    
    model = UNet(in_channels=n_input, n_classes=n_classes, norm="group", up_mode="upconv", depth=depth, wf=wf, padding=True).cuda()
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    
    val_data = MVTecDataset(root=main_path, transform = data_transform, gt_transform=gt_transform, phase='test', dirs = dirs, data_source=args.experiment_name)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size = args.bs, shuffle = False)
    
    run_name = args.experiment_name + '_' +str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_" + args.model + "_" + args.process_method
    
    
    epoch = 'test'
    
    with torch.no_grad():
        model.eval()
        evaluation(args, model, val_dataloader, epoch, loss_l1, visualizer, run_name)