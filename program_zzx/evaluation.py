import torch
import numpy as np
import nibabel as nib
from sklearn import metrics
from loss import FocalLoss, SSIM
from torchvision.utils import save_image


def cal_distance_map(input, target):
    # input = np.squeeze(input, axis=0)
    # target = np.squeeze(target, axis=0)
    d_map = np.full_like(input, 0)
    d_map = np.square(input - target)
    return d_map


def evaluation3D(args, epoch, device, model, test_dataloader, visualizer):
    # model.eval()
    pixel_pred_list, sample_pred_list, gt_list, label_list = [], [], [], []
    error_list = []
    lossMSE = torch.nn.MSELoss()
    with torch.no_grad():
        for (img, img_path) in test_dataloader:
            img = img.to(device)
            x = torch.unsqueeze(img, dim=1)
            # pred = model(x)[1][:,0,:,:,:]
            pred = model(x)[:,0,:,:,:]
            
            #compute error
            error = lossMSE(img, pred)
            error_list.append(error.item())
            
            difference = cal_distance_map(img[0].to('cpu').detach().numpy(), pred[0].to('cpu').detach().numpy())
            
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
        
    print('*'*30)
    print('error:{}'.format(np.mean(error_list)))
    pixelAP = metrics.average_precision_score(gt_list, pixel_pred_list)
    sampleAP = metrics.average_precision_score(label_list, sample_pred_list)

    return pixelAP, sampleAP


def evaluation2D(args, epoch, device, model, test_dataloader, visualizer):
    model.eval()
    pixel_pred_list, sample_pred_list, gt_list, label_list = [], [], [], []
    error_list = []
    lossMSE = torch.nn.MSELoss()
    with torch.no_grad():
        for img, img_path in test_dataloader:
            img = img.to(device)
            outputs = torch.zeros_like(img)  
            
            for i in range(img.shape[2]):
                raw = img[:,i,:,:]
                raw = torch.unsqueeze(raw, dim=1)
                img_slice = img[:,i,:,:]
                img_slice = torch.unsqueeze(img_slice, dim=1)

                output_slice = model(img_slice)
                # output_slice = torch.squeeze(output_slice, dim=1)
                outputs[:,i,:,:] = output_slice
                
                error =  lossMSE(raw, output_slice)
                error_list.append(error.item())
                
            difference = cal_distance_map(img[0].to('cpu').detach().numpy(), outputs[0].to('cpu').detach().numpy())
            
            img_path = img_path[0]
            pixelPath = img_path.replace('toy/', 'toy_label/pixel/')
            samplePath = img_path.replace('toy/', 'toy_label/sample/').replace('nii.gz', 'nii.gz.txt')
            
            pixelGT = nib.load(pixelPath)
            pixelGT = np.rint(pixelGT.get_fdata()).astype(np.int)
            pixelGT = np.transpose(pixelGT, (2, 0, 1))
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
            visualizer.visualize_image_batch(outputs[0,50], epoch, image_name='test_out_50')
            visualizer.visualize_image_batch(outputs[0,125], epoch, image_name='test_out_125')
            visualizer.visualize_image_batch(outputs[0,200], epoch, image_name='test_out_200')     
                
        print('error:{}'.format(np.mean(error_list)))
        
        
    pixelAP = metrics.average_precision_score(gt_list, pixel_pred_list)
    sampleAP = metrics.average_precision_score(label_list, sample_pred_list)

    return pixelAP, sampleAP


def evaluationDRAEM(args, epoch, device, model_rec, model_seg, test_dataloader, visualizer):
    model_rec.eval()
    model_seg.eval()
    pixel_pred_list, sample_pred_list, gt_list, label_list = [], [], [], []
    loss_l2 = torch.nn.modules
    loss_focal = FocalLoss()
    error_list = []
    with torch.no_grad():
        for img, img_path in test_dataloader:
            img = img.to(device)
            
            img_path = img_path[0]
            pixelPath = img_path.replace('toy/', 'toy_label/pixel/')
            samplePath = img_path.replace('toy/', 'toy_label/sample/').replace('nii.gz', 'nii.gz.txt')
            
            pixelGT = nib.load(pixelPath)
            pixelGT = np.rint(pixelGT.get_fdata()).astype(np.int)
            
            with open(samplePath, "r") as val_fl:
                val_str = val_fl.readline()
            sampleGT = int(val_str) 
            
            
            for i in range(img.shape[2]):
                has_anomaly = 1 if pixelGT[:,:,i].sum() > 0 else 0
                # anomaly_mask = torch.tensor(pixelGT[:,:,i]).unsqueeze(dim = 0).unsqueeze(dim = 0).to(device)
                anomaly_mask = torch.tensor(pixelGT[:,:,i])
                anomaly_mask = anomaly_mask.reshape(1, 1, anomaly_mask.shape[0],anomaly_mask.shape[1])
                augmented_image = img[:,i,:,:].unsqueeze(dim = 1)
                
                augmented_image = augmented_image.to(device)
                anomaly_mask = anomaly_mask.to(device)

                image_rec = model_rec(augmented_image)
                joined_in = torch.cat((image_rec, augmented_image), dim=1)

                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)

                segment_loss = loss_focal(out_mask_sm, anomaly_mask)
                error = segment_loss

                error_list.append(error.item())
                
                save_image(augmented_image, 'augmented_image.png')
                save_image(image_rec, 'image_rec.png')
                save_image(out_mask_sm[:, 1:, :, :], 'out_mask_sm.png')
                save_image(anomaly_mask.float(), 'anomlay_mask.png')
                
                difference = out_mask_sm[:, 1:, :, :].to('cpu').detach().numpy()
                
                # pixel
                pixel_pred_list.extend(difference.ravel())
                gt_list.extend(anomaly_mask.to('cpu').detach().numpy().ravel())
                # sample
                sample_pred_list.append(np.sum(difference))
                label_list.append(has_anomaly)
               
                
                assert len(pixel_pred_list) == len(gt_list), "the length of gt and pred don't match!!!"
                if i in [50, 125, 200]:
                    visualizer.visualize_image_batch(img[0,i], epoch, image_name='DRAEM_test_aug_{}'.format(i))
                    visualizer.visualize_image_batch(image_rec, epoch, image_name='DRAEM_test_rec_{}'.format(i))
                    visualizer.visualize_image_batch(out_mask_sm, epoch, image_name='DRAEM_test_seg_{}'.format(i))
                    visualizer.visualize_image_batch(anomaly_mask, epoch, image_name='DRAEM_test_mask_{}'.format(i))
        print('error:{}'.format(np.mean(error_list)))
        
        
    pixelAP = metrics.average_precision_score(gt_list, pixel_pred_list)
    sampleAP = metrics.average_precision_score(label_list, sample_pred_list)

    return pixelAP, sampleAP