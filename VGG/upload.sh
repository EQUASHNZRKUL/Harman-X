#!/bin/bash

scp ./VGG.py exx@10.35.120.26:Desktop/SummerDesktop/ctg_dsp_share/justin_data
scp ./VGG_test.py exx@10.35.120.26:Desktop/SummerDesktop/ctg_dsp_share/justin_data
scp ./VGG_train.py exx@10.35.120.26:Desktop/SummerDesktop/ctg_dsp_share/justin_data
scp -r ./summary/ exx@10.35.120.26:Desktop/SummerDesktop/ctg_dsp_share/justin_data
scp -r ./checkpoints/ exx@10.35.120.26:Desktop/SummerDesktop/ctg_dsp_share/justin_data
