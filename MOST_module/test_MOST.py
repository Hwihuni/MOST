import argparse
import logging
import os
import scipy.io
import hdf5storage
import numpy as np
import torch
import torch.nn as nn
from unet import *
from tqdm import tqdm
from torch.utils.data import DataLoader
import copy
from utils import  init_weights_zeros
from var_models import VarNet, Unet



def predict_net(recon_net,task_net,args,val_loader):

    recon_net.eval()
    if task_net is not None:
        task_net.eval()
    n_val = len(val)
    
    recons = []
    labels = []
    for imgs, kspace, mask, fimg, target in tqdm(val_loader):
        imgs = imgs.cuda(args.gpu_ind[0])
        kspace = kspace.cuda(args.gpu_ind[0])
        mask = mask.cuda(args.gpu_ind[0]).byte()
        target = target.cuda(args.gpu_ind[0])
        
        with torch.no_grad():            
            if args.task.endswith('class') or args.task.endswith('pred'):
                img_shape = imgs.shape
                mask = mask[0:1,:,:,:]
                imgs = torch.reshape(imgs,(img_shape[0]*img_shape[1],1,img_shape[2],img_shape[3]))
                kspace = torch.reshape(kspace,(img_shape[0]*img_shape[1],1,img_shape[2],img_shape[3],2))

            dc_output = nn.parallel.data_parallel(recon_net, (kspace, kspace, mask), args.gpu_ind)
            imgs_pred = dc_output
                
            if args.task.endswith('class') or args.task.endswith('pred'):
                imgs_pred = torch.reshape(imgs_pred,(img_shape[0],img_shape[1],img_shape[2],img_shape[3])).permute(0,2,3,1).unsqueeze(1)
            if not args.task.endswith('recon'):
                preds = nn.parallel.data_parallel(task_net, imgs_pred, args.gpu_ind)
            #if args.task.endswith('class'):                
                #preds = preds > 0
        
        for i in range(target.shape[0]):
            label = torch.squeeze(target[i,...]).cpu().detach().numpy()
            if args.task.endswith('recon'):
                pred = torch.squeeze(dc_output[i,...]).cpu().detach().numpy()
            else:
                pred = torch.squeeze(preds[i,...]).cpu().detach().numpy()
                
            recons.append(pred)
            labels.append(label)
        
    return recons, labels

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=1000,help='Number of epochs', dest='epochs')
    parser.add_argument('--lr', type=float, nargs='?', default=1e-3,help='Learning rate', dest='lr')
    parser.add_argument('-da', '--task', dest='task', type=str, default='Task1_OASIS1_Tissue_seg',help='dataset') 
    parser.add_argument('-gi', '--gpu_ind', dest='gpu_ind', type=str, default='2',help='gpu')
    parser.add_argument('-vs', '--valid_step', dest='valid_step', type=int, default=1,help='Validation round step')
    parser.add_argument('-ss', '--save_step', dest='save_step', type=int, default=1,help='Checkpoint saving step')
    parser.add_argument('--acceleration', dest='acceleration', type=float, default=4,help='acceleration')
    parser.add_argument('--center_fraction', dest='center_fraction', type=float, default=0.08,help='center_fraction')
    parser.add_argument('-cn', '--class_num', type=int, default=1,help='class number', dest='class_num')
    parser.add_argument('-ms', '--buffer_step', dest='buffer_step', type=int, default=3,help='buffer step')
    parser.add_argument('--buffer_size',  type=int, default=10,help='Buffer size')

    return parser.parse_args()

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    str_ids = args.gpu_ind.split(',')
    args.gpu_ind = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ind.append(id)

    # set gpu ids
    if len(args.gpu_ind) > 0:
        torch.cuda.set_device(args.gpu_ind[0])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device CUDA {(args.gpu_ind)}')


    dir_name = 'inf_metric'
    os.makedirs(dir_name, exist_ok=True)
    logging.info(f'directory named {dir_name} is made')
    
    recon_net = VarNet(num_cascades=6, pools=4, chans=18, sens_pools=4, sens_chans=8)
    recon_net.cuda()
    
    if args.task.endswith('seg'):
        from dataset.dataset_vn_cl_seg import BasicDataset
        from unet import UNet
        metrics = ['IoU','DICE']
        def metr_cal1(x,y):
            return np.nanmean([np.sum((a>0)*(b>0))/np.sum((a>0)+(b>0)>0) for (a,b) in zip(x,y)])
        def metr_cal2(x,y):
            return np.nanmean([2*np.sum((a>0)*(b>0))/(np.sum((a>0))+np.sum((b>0))) for (a,b) in zip(x,y)])

        direc = 'Segmentation'
        task_net = UNet(n_channels=1, n_classes=args.class_num, bilinear=True)
        task_net .to(device=device)
        chck = f'./checkpoints/downstream_task/{args.task[:5]}.pth'
        print(chck)
        task_net.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
        args.batchsize = 100
         
    elif args.task.endswith('class'):
        from dataset.dataset_vn_cl_class import BasicDataset
        from Resnet import Resnet
        metrics = ['Acc','AUC']
        
        def metr_cal1(x,y):
            return np.mean([((a>0)==b) for (a,b) in zip(x,y)])
        def metr_cal2(x,y):
            auc = 0
            TP = 1
            FP = 1
            x_sort = np.sort(x)
            for ind in range(len(x)):
                TP_old = TP
                FP_old = FP
                th = x_sort[ind]
                TP = np.sum([((a>th)  *(b==1)) for (a,b) in zip(x,y)])/np.sum(y)
                FP = np.sum([((a>th)  *(b==0)) for (a,b) in zip(x,y)])/(len(y)-np.sum(y))
                #auc += np.mean([((a>(np.log(0.01*th/(1-0.01*th))))  *(b==1)) for (a,b) in zip(x,y)])
                auc += (TP_old + TP)*(FP_old-FP)/2
            auc += (TP)*(FP)/2
            return auc

        direc = 'Classification'
        if args.task == 'Task4_IXI-T1_Sex_class':
            flattened_shape=[-1, 512, 5, 7, 2]
        elif args.task == 'Task5_ADNI_ADCN_class':
            flattened_shape = [-1, 512, 6, 8, 2]
        task_net = Resnet(n_classes = 1,flattened_shape=flattened_shape,dropout = 0)
        
        task_net.to(device=device)

        chck = f'./checkpoints/downstream_task/{args.task[:5]}.pth'
        logging.info(chck)
        task_net.load_state_dict(torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}')))
        args.batchsize = 5
        
        
    else:
        from dataset.dataset_vn import BasicDataset
        metrics = ['PSNR','SSIM']
        from skimage.metrics import structural_similarity
        from skimage.metrics import peak_signal_noise_ratio
        def metr_cal1(x,y):
            return np.mean([peak_signal_noise_ratio(a,b,data_range = np.max(b)) for (a,b) in zip(x,y)])
        def metr_cal2(x,y):
            return np.mean([structural_similarity(a,b) for (a,b) in zip(x,y)])
        task_net = None
        args.batchsize = 100
        
    val = BasicDataset(args,'test')
    val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    

    task_list = ['Task0_Fastmri_recon','Task1_OASIS1_Tissue_seg','Task3_BRATS_Tumor_seg','Task4_IXI-T1_Sex_class','Task5_ADNI_ADCN_class']    #task_ind_start = task_list.index(args.task)
    task_ind_start = 0
    
    method = args.method


    metric1_means = []
    metric2_means = []
    for task_ind in range(task_ind_start,len(task_list)):
        task_trained = task_list[task_ind]
            
        if task_trained == 'Task0_Fastmri_recon':
            chck = f'./checkpoints/pretrained_recon/reconnet_epoch_best.pth' # Pretrained recon network checkpoint
        else:
            chck = f'./checkpoints/Trn_MOST_buffer_size{args.buffer_size}_step{args.buffer_step}_chk/{task_trained}/recon_net_step_best.pth'
        print(chck)
        
        model_dict = recon_net.state_dict() # 현재 신경망 상태 로드 
        pretrained_dict = torch.load(chck,map_location=torch.device(f'cuda:{args.gpu_ind[0]}'))  # pretrained 상태 로드

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        recon_net.load_state_dict(pretrained_dict)

        args.task_trained = task_trained
        recons, labels  = predict_net(recon_net,task_net,args,val_loader)
        met1 = metr_cal1(recons,labels)
        met2 = metr_cal2(recons,labels)
        

        
        print(f'{method}: {metrics[0]}: {met1}, {metrics[1]}: {met2}')
        metric1_means.append(np.mean(met1))
        metric2_means.append(np.mean(met2))
        
        print(f'{method}: {metrics[0]} mean: {np.mean(metric1_means)}, {metrics[1]} mean: {np.mean(metric2_means)}')
            
        # np.save(f'inf_metric/{method}_{metrics[0]}_{args.task}_means.npy',metric1_means)
        # np.save(f'inf_metric/{method}_{metrics[1]}_{args.task}_means.npy',metric2_means)
        
        if task_ind != task_ind_start:
            if args.task.endswith('pred'):
                print(f'{metrics[0]} forget: {metric1_means[-1]-np.min(metric1_means)}, {metrics[1]} forget: {metric2_means[-1]-np.min(metric2_means)}')
            else:
                print(f'{metrics[0]} forget: {np.max(metric1_means)-metric1_means[-1]}, {metrics[1]} forget: {np.max(metric2_means)-metric2_means[-1]}')
                
