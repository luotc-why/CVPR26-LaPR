#!/bin/bash

mkdir -p logs
currenttime=$(date "+%Y%m%d_%H%M%S")

declare -A GPU_MAP=(
    [0]=0
    [1]=1
    [2]=2
    [3]=3
)

for folderid in 0 1 2 3; do
  for seed in 0; do
    for lr in 0.005; do
      gpu_id=${GPU_MAP[$folderid]}
      log_file="/data1/liyusheng/CVPR-LRKL/Data/logs_MLP_MoE/${folderid}-${seed}-${lr}-${currenttime}.log"

      echo "Launching folder_id=${folderid}, seed=${seed}, lr=${lr} on GPU ${gpu_id} ..."

      CUDA_VISIBLE_DEVICES=${gpu_id} python main_supcon_MLP_MoE.py \
        --batch_size 64 \
        --learning_rate ${lr} \
        --epochs 200 \
        --temp 0.1 \
        --dataset path \
        --data_base_path /data1/liyusheng/CVPR-LRKL/Data \
        --data_folder /data1/liyusheng/CVPR-LRKL/Data/figures_dataset/pascal-5i/VOC2012/JPEGImages \
        --seed ${seed} \
        --folder_id ${folderid} \
        --cosine \
        --pretrain \
        2>&1 | tee "${log_file}" > /dev/null &

      echo -e "\033[32m[ Running: ${log_file} ]\033[0m"
    done
  done
done

wait
echo -e "\033[34mAll jobs completed. Logs in ./logs\033[0m"