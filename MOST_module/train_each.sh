#!/bin/bash

GPU=0
start=dc
arch=dc

nohup python train_each.py --task Task1_OASIS1_Tissue_seg -l 0.001 --gpu_ind $GPU --start $start --arch $arch
nohup python train_each.py --task Task3_BRATS_Tumor_seg -l 0.0001 --gpu_ind $GPU --start $start --arch $arch
nohup python train_each.py --task Task4_IXI-T1_Sex_class -l 0.0001 --gpu_ind $GPU --start $start --arch $arch
nohup python train_each.py --task Task5_ADNI_ADCN_class -l 0.0001 --gpu_ind $GPU --start $start --arch $arch




mode=alltask

method=each_start_dc_arch_dc
file=$dir$method$out 


task=Task0_Fastmri_recon
nohup python predict_forget_dconly_$mode.py --task $task --gpu_ind $GPU --method $method 
task=Task1_OASIS1_Tissue_seg
nohup python predict_forget_dconly_$mode.py --task $task --gpu_ind $GPU --method $method 
task=Task3_BRATS_Tumor_seg
nohup python predict_forget_dconly_$mode.py --task $task --gpu_ind $GPU --method $method 
task=Task4_IXI-T1_Sex_class
nohup python predict_forget_dconly_$mode.py --task $task --gpu_ind $GPU --method $method
task=Task5_ADNI_ADCN_class
nohup python predict_forget_dconly_$mode.py --task $task --gpu_ind $GPU --method $method


