export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc-per-node 4 train.py \
    --model_path Initial-Vi-Llama \
    --local_data_path vi-wiki
