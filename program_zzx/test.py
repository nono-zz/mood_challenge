import torch
import numpy as np
import nibabel as nib
from sklearn import metrics


def evaluation(args, epoch, device, model, test_dataloader):
    model.eval()
    pixel_pred_list, sample_pred_list, gt_list, label_list = [], []
    for img, img_path in test_dataloader:
        img.to(device)
        pred = model(img)
        
        
        pixelPath = img_path.replace('toy', 'toy_label/pixel')
        samplePath = img_path.replace('toy', 'toy_label/sample')
        
        pixelGT = nib.load(pixelPath)
        with open(samplePath, "r") as val_fl:
            val_str = val_fl.readline()
        sampleGT = int(val_str) 
        
        pred_array = pred.to('cpu').detach().numpy()
        pixel_pred_list.extend(pred_array.ravel())
        sample_pred_list.append(np.sum(pred_array))
        label_list.append(sampleGT)
        gt_list.extend(pixelGT.detach().numpy())
        
        
        pixelAP = metrics.average_precision_score(pixel_pred_list, gt_list)
        sampleAP = metrics.average_precision_score(pixel_pred_list, label_list)

        return pixelAP, sampleAP