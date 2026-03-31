#!/bin/bash

mkdir -p logs
currenttime=$(date "+%Y%m%d_%H%M%S")

feature_name=$1
output_name=$2

declare -A GPU_MAP=(
    [0]=4
    [1]=5
    [2]=6
    [3]=7
)

for folderid in 0 1 2 3; do
    for seed in 0; do
        gpu_id=${GPU_MAP[$folderid]}
        log_file="/data1/liyusheng/CVPR-LRKL/Data/logs_XY/${folderid}-${seed}-${currenttime}.log"
        echo "Launching fold=${folderid} (GPU ${gpu_id}) in background..."

        CUDA_VISIBLE_DEVICES=${gpu_id} python evaluate/evaluate_segmentation.py \
            --model mae_vit_large_patch16 \
            --base_dir /data1/liyusheng/CVPR-LRKL/Data/ \
            --feature_name "${feature_name}" \
            --output_dir "/data1/liyusheng/CVPR-LRKL/Data/output_seg_images/moe_XY_segmentation/${output_name}_folder${folderid}_seed${seed}" \
            --ckpt /data1/liyusheng/CVPR-LRKL/Data/weights/checkpoint-1000.pth \
            --fold ${folderid} \
            --seed ${seed} \
            2>&1 | tee "${log_file}" > /dev/null &

        echo -e "\033[32m[ Running fold=${folderid}, log: ${log_file} ]\033[0m"
    done
done

wait
echo -e "\033[34mAll folds completed. Logs saved in ./logs/\033[0m"
