#!/bin/bash

# cd /home/tanner/embodiedAI_ws/starVLA
# conda activate starVLA

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/home/tanner/embodiedAI_ws/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/home/tanner/anaconda3/envs/libero/bin/python

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo


host="127.0.0.1"
base_port=5694
unnorm_key="franka"
your_ckpt=/home/tanner/embodiedAI_ws/starVLA/results/Checkpoints/1218_libero_all_qwen3oft/checkpoints/steps_30000_pytorch_model.pt
# export DEBUG=true

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}

num_trials_per_task=50

# 定义所有要评估的task suites
# task_suites=("libero_spatial" "libero_object" "libero_goal" "libero_10")
task_suites=("libero_90")

# 循环运行每个task suite
for task_suite_name in "${task_suites[@]}"; do
    echo "=========================================="
    echo "Starting evaluation for: $task_suite_name"
    echo "=========================================="
    
    video_out_path="results/${task_suite_name}/${folder_name}"
    
    ${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
        --args.pretrained-path ${your_ckpt} \
        --args.host "$host" \
        --args.port $base_port \
        --args.task-suite-name "$task_suite_name" \
        --args.num-trials-per-task "$num_trials_per_task" \
        --args.video-out-path "$video_out_path" \
        --args.unnorm_key $unnorm_key \
        2>&1 | tee "${LOG_DIR}/${task_suite_name}.log"
    
    echo "Completed: $task_suite_name"
    echo ""
done

echo "=========================================="
echo "All evaluations completed!"
echo "Results saved in: results/"
echo "Logs saved in: ${LOG_DIR}"
echo "=========================================="