#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export star_vla_python=/home/tanner/anaconda3/envs/starVLA/bin/python
# your_ckpt=/home/tanner/embodiedAI_ws/starVLA/results/Checkpoints/1220_custom_datasets_mix_libero_qwen3oft/final_model/pytorch_model.pt
your_ckpt=/home/tanner/embodiedAI_ws/starVLA/results/Checkpoints/1218_libero_all_qwen3oft/checkpoints/steps_30000_pytorch_model.pt
gpu_id=0
port=5694
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
