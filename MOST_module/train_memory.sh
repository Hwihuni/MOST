#!/bin/bash


GPU=1
memory_size=4
memory_step=3

nohup python train_cl_start_dc_arch_dc_taskloss_memory.py --task Task1_OASIS1_Tissue_seg -l 0.001 --gpu_ind $GPU --memory_step $memory_step --memory_size $memory_size
nohup python train_cl_start_dc_arch_dc_taskloss_memory.py --task Task3_BRATS_Tumor_seg -l 0.0001 --gpu_ind $GPU --memory_step $memory_step --memory_size $memory_size --epochs 3 
nohup python train_cl_start_dc_arch_dc_taskloss_memory.py --task Task4_IXI-T1_Sex_class -l 0.0001 --gpu_ind $GPU --memory_step $memory_step --memory_size $memory_size
nohup python train_cl_start_dc_arch_dc_taskloss_memory.py --task Task5_ADNI_ADCN_class -l 0.0001 --gpu_ind $GPU --memory_step $memory_step --memory_size $memory_size




dir=nohupfile/
out=.out
echo $file


mode=alltask

method=start_dc_arch_dc_taskloss_memory4_step3
file=$dir$method$out 


task=Task0_Fastmri_recon
nohup python predict_forget_dconly_$mode.py --task $task --gpu_ind $GPU --method $method >> $file
task=Task1_OASIS1_Tissue_seg
nohup python predict_forget_dconly_$mode.py --task $task --gpu_ind $GPU --method $method >> $file
task=Task3_BRATS_Tumor_seg
nohup python predict_forget_dconly_$mode.py --task $task --gpu_ind $GPU --method $method >> $file
task=Task4_IXI-T1_Sex_class
nohup python predict_forget_dconly_$mode.py --task $task --gpu_ind $GPU --method $method >> $file
task=Task5_ADNI_ADCN_class
nohup python predict_forget_dconly_$mode.py --task $task --gpu_ind $GPU --method $method >> $file

#!/bin/bash
