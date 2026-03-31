#!/bin/bash
set -e

data_root=$1

# [1]
python tools/featextrater_folderwise_SupMoEPR_XY.py vit_large_patch14_clip_224.laion2b features_supcon-vit-laion2b-clip-csz224-bsz64-lr0005-freeze-encoder trn ${data_root} 0.005 ckpt_epoch_200 1 "0,1,2,3" 
python tools/featextrater_folderwise_SupMoEPR_XY.py vit_large_patch14_clip_224.laion2b features_supcon-vit-laion2b-clip-csz224-bsz64-lr0005-freeze-encoder val ${data_root} 0.005 ckpt_epoch_200 1 "0,1,2,3" 

# [2]
python tools/calculate_similariity_SupMoE_XY.py features_supcon-vit-laion2b-clip-csz224-bsz64-lr0005-freeze-encoder val trn ${data_root} 0.005 ckpt_epoch_200 1 "0,1,2,3" 

# [3]
python evaluate/evaluate_segmentation.py --model mae_vit_large_patch16 --base_dir ${data_root} --feature_name features_supcon-vit-laion2b-clip-csz224-bsz64-lr0005-freeze-encoder_val --output_dir ${data_root}/cross_1_output_seg_images/moe_moe_segmentation_epoch-200/output_supcon-vit-laion2b-clip-csz224-bsz64-lr0005-freeze-encoder_val_XY_folder0_seed0 --ckpt ${data_root}/weights/checkpoint-3400.pth --fold 0 --seed 0 --top_50_path ${data_root}/figures_dataset/pascal-5i/VOC2012/features_supcon-vit-laion2b-clip-csz224-bsz64-lr0005-freeze-encoder_val_MoE_final_XY_lr_0.005_epoch_ckpt_epoch_200_from_folder1/folder0_top50-similarity.json 

python evaluate/evaluate_segmentation.py --model mae_vit_large_patch16 --base_dir ${data_root} --feature_name features_supcon-vit-laion2b-clip-csz224-bsz64-lr0005-freeze-encoder_val --output_dir ${data_root}/cross_1_output_seg_images/moe_moe_segmentation_epoch-200/output_supcon-vit-laion2b-clip-csz224-bsz64-lr0005-freeze-encoder_val_XY_folder1_seed0 --ckpt ${data_root}/weights/checkpoint-3400.pth --fold 1 --seed 0 --top_50_path ${data_root}/figures_dataset/pascal-5i/VOC2012/features_supcon-vit-laion2b-clip-csz224-bsz64-lr0005-freeze-encoder_val_MoE_final_XY_lr_0.005_epoch_ckpt_epoch_200_from_folder1/folder1_top50-similarity.json 

python evaluate/evaluate_segmentation.py --model mae_vit_large_patch16 --base_dir ${data_root} --feature_name features_supcon-vit-laion2b-clip-csz224-bsz64-lr0005-freeze-encoder_val --output_dir ${data_root}/cross_1_output_seg_images/moe_moe_segmentation_epoch-200/output_supcon-vit-laion2b-clip-csz224-bsz64-lr0005-freeze-encoder_val_XY_folder2_seed0 --ckpt ${data_root}/weights/checkpoint-3400.pth --fold 2 --seed 0 --top_50_path ${data_root}/figures_dataset/pascal-5i/VOC2012/features_supcon-vit-laion2b-clip-csz224-bsz64-lr0005-freeze-encoder_val_MoE_final_XY_lr_0.005_epoch_ckpt_epoch_200_from_folder1/folder2_top50-similarity.json 

python evaluate/evaluate_segmentation.py --model mae_vit_large_patch16 --base_dir ${data_root} --feature_name features_supcon-vit-laion2b-clip-csz224-bsz64-lr0005-freeze-encoder_val --output_dir ${data_root}/cross_1_output_seg_images/moe_moe_segmentation_epoch-200/output_supcon-vit-laion2b-clip-csz224-bsz64-lr0005-freeze-encoder_val_XY_folder3_seed0 --ckpt ${data_root}/weights/checkpoint-3400.pth --fold 3 --seed 0 --top_50_path ${data_root}/figures_dataset/pascal-5i/VOC2012/features_supcon-vit-laion2b-clip-csz224-bsz64-lr0005-freeze-encoder_val_MoE_final_XY_lr_0.005_epoch_ckpt_epoch_200_from_folder1/folder3_top50-similarity.json 

