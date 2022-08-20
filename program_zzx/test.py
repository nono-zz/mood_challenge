import torch
import numpy as np
import nibabel as nib
from sklearn import metrics


def cal_distance_map(input, target):
    # input = np.squeeze(input, axis=0)
    # target = np.squeeze(target, axis=0)
    d_map = np.full_like(input, 0)
    d_map = np.square(input - target)
    return d_map


def evaluation(args, epoch, device, model, test_dataloader, visualizer):
    model.eval()
    pixel_pred_list, sample_pred_list, gt_list, label_list = [], [], [], []
    for (img, img_path) in test_dataloader:
        img = img.to(device)
        x = torch.unsqueeze(img, dim=1)
        pred = model(x)[1][:,0,:,:,:]
        
        difference = cal_distance_map(img[0,:,:].to('cpu').detach().numpy(), pred[0,:,:].to('cpu').detach().numpy())
        
        img_path = img_path[0]
        pixelPath = img_path.replace('toy/', 'toy_label/pixel/')
        samplePath = img_path.replace('toy/', 'toy_label/sample/').replace('nii.gz', 'nii.gz.txt')
        
        pixelGT = nib.load(pixelPath)
        pixelGT = np.rint(pixelGT.get_fdata()).astype(np.int)
        with open(samplePath, "r") as val_fl:
            val_str = val_fl.readline()
        sampleGT = int(val_str) 
        
        # pred_array = pred.to('cpu').detach().numpy()
        pixel_pred_list.extend(difference.ravel())
        sample_pred_list.append(np.sum(difference))
        label_list.append(sampleGT)
        gt_list.extend(pixelGT.ravel())
        
        assert len(pixel_pred_list) == len(gt_list), "the length of gt and pred don't match!!!"
        
        
        visualizer.visualize_image_batch(img[0,50], epoch, image_name='test_img_50')
        visualizer.visualize_image_batch(img[0,125], epoch, image_name='test_img_125')
        visualizer.visualize_image_batch(img[0,200], epoch, image_name='test_img_200')
        visualizer.visualize_image_batch(pred[0,50], epoch, image_name='test_out_50')
        visualizer.visualize_image_batch(pred[0,125], epoch, image_name='test_out_125')
        visualizer.visualize_image_batch(pred[0,200], epoch, image_name='test_out_200')    
        
        
    pixelAP = metrics.average_precision_score(gt_list, pixel_pred_list)
    sampleAP = metrics.average_precision_score(label_list, sample_pred_list)

    return pixelAP, sampleAP