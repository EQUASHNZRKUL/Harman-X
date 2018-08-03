#!/bin/bash

clear

ls
cd ~/Documents/TensorFlowPractice/VGG

echo "Starting upload..."

echo "Uploading VGG.py..."
scp VGG.py exx@10.35.120.26:Desktop/SummerDesktop/ctg_dsp_share/justin_data

echo "Uploading VGG_test.py..."
scp VGG_test.py exx@10.35.120.26:Desktop/SummerDesktop/ctg_dsp_share/justin_data

echo "Uploading VGG_train.py..."
scp VGG_train.py exx@10.35.120.26:Desktop/SummerDesktop/ctg_dsp_share/justin_data

echo "done!"

# scp VGG.py exx@10.35.120.26:Desktop/SummerDesktop/ctg_dsp_share/justin_data
# scp VGG_test.py exx@10.35.120.26:Desktop/SummerDesktop/ctg_dsp_share/justin_data
# scp VGG_train.py exx@10.35.120.26:Desktop/SummerDesktop/ctg_dsp_share/justin_data
# scp -r summary/ exx@10.35.120.26:Desktop/SummerDesktop/ctg_dsp_share/justin_data
# scp -r checkpoints/ exx@10.35.120.26:Desktop/SummerDesktop/ctg_dsp_share/justin_data

