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

echo "Uploading ./summary/..."
scp -r summary/ exx@10.35.120.26:Desktop/SummerDesktop/ctg_dsp_share/justin_data

echo "Uploading ./checkpoints/..."
scp -r checkpoints/ exx@10.35.120.26:Desktop/SummerDesktop/ctg_dsp_share/justin_data

echo "Uploading ./MFCCData/..."
scp -r ../FileFinderFolder/PSF/MFCCData_folder/MFCCData_split exx@10.35.120.26:Desktop/SummerDesktop/ctg_dsp_share/justin_data/MFCCData_folder/MFCCData_split
scp ../FileFinderFolder/PSF/MFCCData_folder/MFCCData.zip exx@10.35.120.26:Desktop/SummerDesktop/ctg_dsp_share/justin_data/MFCCData_folder

echo "done!"

# scp VGG.py exx@10.35.120.26:Desktop/SummerDesktop/ctg_dsp_share/justin_data
# scp VGG_test.py exx@10.35.120.26:Desktop/SummerDesktop/ctg_dsp_share/justin_data
# scp VGG_train.py exx@10.35.120.26:Desktop/SummerDesktop/ctg_dsp_share/justin_data
# scp -r summary/ exx@10.35.120.26:Desktop/SummerDesktop/ctg_dsp_share/justin_data
# scp -r checkpoints/ exx@10.35.120.26:Desktop/SummerDesktop/ctg_dsp_share/justin_data