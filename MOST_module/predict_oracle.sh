#!/bin/bash

GPU=0
file=predict_oracle.out

nohup python predict_oracle.py --task Task0_Fastmri_recon --gpu_ind $GPU  >> $file
nohup python predict_oracle.py --task Task1_OASIS1_Tissue_seg --gpu_ind $GPU  >> $file
nohup python predict_oracle.py --task Task3_BRATS_Tumor_seg  --gpu_ind $GPU  >> $file
nohup python predict_oracle.py --task Task4_IXI-T1_Sex_class --gpu_ind $GPU  >> $file
nohup python predict_oracle.py --task Task5_ADNI_ADCN_class --gpu_ind $GPU  >> $file