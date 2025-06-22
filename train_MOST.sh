#!/bin/bash


GPU=0
buffer_size=4
buffer_step=3
cd MOST_module
nohup python train_MOST.py --task Task1_OASIS1_Tissue_seg -l 0.001 --gpu_ind $GPU --buffer_step $buffer_step --buffer_size $buffer_size
nohup python train_MOST.py --task Task3_BRATS_Tumor_seg -l 0.0001 --gpu_ind $GPU --buffer_step $buffer_step --buffer_size $buffer_size  
nohup python train_MOST.py --task Task4_IXI-T1_Sex_class -l 0.0001 --gpu_ind $GPU --buffer_step $buffer_step --buffer_size $buffer_size
nohup python train_MOST.py --task Task5_ADNI_ADCN_class -l 0.0001 --gpu_ind $GPU --buffer_step $buffer_step --buffer_size $buffer_size



