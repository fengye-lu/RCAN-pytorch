#!/bin/bash/
# For release
# RCAN_BIX2
CUDA_VISIBLE_DEVICES=0 /home/shiyanshi/下载/BestBuyBig/venv/bin/python main.py --data_test Urban100 --scale 2 --model RCAN --n_resgroups 10 --n_resblocks 6 --n_feats 64 --pre_train /home/shiyanshi/项目代码/卢正浩/RCAN-pytorch/RCAN_TrainCode/experiment/RCAN_BIX2_G5R20P48/model/model_best.pt --test_only --save_results --chop --save 'RCAN-x2-test'
#UDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train /home/shiyanshi/项目代码/卢正浩/RCAN-pytorch/RCAN_TrainCode/experiment/RCAN_BIX2_G10R20P48/model/model_best.pt --test_only --save_results --chop --save 'RCAN' --testpath /home/shiyanshi/项目代码/卢正浩/Dataset --testset taiheiziphoto
# RCAN_BIX3
#CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX3.pt --test_only --save_results --chop --save 'RCAN' --testpath /media/yulun/Disk10T/datasets/super-resolution/LRBI --testset Set5
# RCAN_BIX4
#CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4 --model RCAN --n_resgroups 2 --n_resblocks 20 --n_feats 64 --pre_train /home/shiyanshi/项目代码/卢正浩/RCAN-pytorch/RCAN_TrainCode/experiment/RCAN_BIX4_G2R20P48/model/model_best.pt --test_only --save_results --chop --save 'RCAN' --testpath /home/shiyanshi/项目代码/卢正浩/Dataset/benchmark --testset Set5
# RCAN_BIX8
#CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 8 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX8.pt --test_only --save_results --chop --save 'RCAN' --testpath /media/yulun/Disk10T/datasets/super-resolution/LRBI --testset Set5
##
# RCANplus_BIX2
#CUDA_VISIBLE_DEVICES=3 python main.py --data_test MyImage --scale 2 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX2.pt --test_only --save_results --chop --self_ensemble --save 'RCANplus' --testpath /media/yulun/Disk10T/datasets/super-resolution/LRBI --testset Set5
# RCANplus_BIX3
#CUDA_VISIBLE_DEVICES=3 python main.py --data_test MyImage --scale 3 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX3.pt --test_only --save_results --chop --self_ensemble --save 'RCANplus' --testpath /media/yulun/Disk10T/datasets/super-resolution/LRBI --testset Set5
# RCANplus_BIX4
#CUDA_VISIBLE_DEVICES=3 python main.py --data_test MyImage --scale 4 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX4.pt --test_only --save_results --chop --self_ensemble --save 'RCANplus' --testpath /media/yulun/Disk10T/datasets/super-resolution/LRBI --testset Set5
# RCANplus_BIX8
#CUDA_VISIBLE_DEVICES=3 python main.py --data_test MyImage --scale 8 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX8.pt --test_only --save_results --chop --self_ensemble --save 'RCANplus' --testpath /media/yulun/Disk10T/datasets/super-resolution/LRBI --testset Set5

