#!/bin/bash

GPU=0

nohup python train_cl_unet.py --task Task1_OASIS1_Tissue_seg -l 0.001 --gpu_ind $GPU
nohup python train_cl_unet.py --task Task3_BRATS_Tumor_seg -l 0.0001 --gpu_ind $GPU --epochs 3 
nohup python train_cl_unet.py --task Task4_IXI-T1_Sex_class -l 0.0001 --gpu_ind $GPU
nohup python train_cl_unet.py --task Task5_ADNI_ADCN_class -l 0.0001 --gpu_ind $GPU


method=unet

dir=nohupfile/
out=.out
file=$dir$method$out 
echo $file

mode=alltask

task=Task0_Fastmri_recon
nohup python predict_forget_unet_$mode.py --task $task --gpu_ind $GPU >> $file
task=Task1_OASIS1_Tissue_seg
nohup python predict_forget_unet_$mode.py --task $task --gpu_ind $GPU >> $file
task=Task3_BRATS_Tumor_seg
nohup python predict_forget_unet_$mode.py --task $task --gpu_ind $GPU >> $file
task=Task4_IXI-T1_Sex_class
nohup python predict_forget_unet_$mode.py --task $task --gpu_ind $GPU >> $file
task=Task5_ADNI_ADCN_class
nohup python predict_forget_unet_$mode.py --task $task --gpu_ind $GPU >> $file


