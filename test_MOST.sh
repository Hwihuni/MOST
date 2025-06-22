#!/bin/bash


GPU=0
buffer_size=4
buffer_step=3

cd MOST_module

file=MOST.out
echo $file 

task=Task0_Fastmri_recon
nohup python test_MOST.py --task $task --gpu_ind $GPU --buffer_step $buffer_step --buffer_size $buffer_size >> $file
task=Task1_OASIS1_Tissue_seg
nohup python test_MOST.py --task $task --gpu_ind $GPU --buffer_step $buffer_step --buffer_size $buffer_size >> $file
task=Task3_BRATS_Tumor_seg
nohup python test_MOST.py --task $task --gpu_ind $GPU --buffer_step $buffer_step --buffer_size $buffer_size >> $file
task=Task4_IXI-T1_Sex_class
nohup python test_MOST.py --task $task --gpu_ind $GPU --buffer_step $buffer_step --buffer_size $buffer_size >> $file
task=Task5_ADNI_ADCN_class
nohup python test_MOST.py --task $task --gpu_ind $GPU --buffer_step $buffer_step --buffer_size $buffer_size >> $file

