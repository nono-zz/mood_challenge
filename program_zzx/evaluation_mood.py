import torch
import os
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


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