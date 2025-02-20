# WITH TMP ACCESS, REMOVE MOST OF THESE
export TMPDIR=/data1/joey/tmp
export TMP=/data1/joey/tmp
export TEMP=/data1/joey/tmp
export TEMPDIR=/data1/joey/tmp
export GCC_TMPDIR=/data1/joey/tmp
export NVCC_TMPDIR=/data1/joey/tmp
#export TORCH_EXTENSIONS_DIR=/data1/joey/torch_extensions
export HOME=/data1/joey
export DS_BUILD_TEMP_DIR=/data1/joey/tmp
export CCACHE_TEMPDIR=/data1/joey/tmp
export HF_HOME=/data1/joey/hf_cache
export HF_HUB_CACHE=/data1/joey/hf_cache
export RAY_BACKEND_LOG_LEVEL=debug
export VLLM_CONFIGURE_LOGGING=0

source .env

set -x

uv run ray job submit --address="http://127.0.0.1:8265" \
  --working-dir . \
  --runtime-env-json='{
  "setup_commands": ["pip install openrlhf[vllm]"], 
  "env_vars": {
    "CUDA_LAUNCH_BLOCKING": "1",
    "TMPDIR": "/data1/joey/tmp", 
    "TMP": "/data1/joey/tmp", 
    "TEMP": "/data1/joey/tmp", 
    "TEMPDIR": "/data1/joey/tmp", 
    "GCC_TMPDIR": "/data1/joey/tmp", 
    "NVCC_TMPDIR": "/data1/joey/tmp", 
    "HOME": "/data1/joey", 
    "DS_BUILD_TEMP_DIR": "/data1/joey/tmp", 
    "CCACHE_TEMPDIR": "/data1/joey/tmp", 
    "HF_HOME": "/data1/joey/hf_cache", 
    "RAY_BACKEND_LOG_LEVEL": "debug"}}' \
  -- python -m openrlhf.cli.train_ppo_ray \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 1 \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 4 \
  --vllm_num_engines 2 \
  --vllm_tensor_parallel_size 1 \
  --pretrain deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --save_path /data1/joey/rl_math/checkpoint/test-rl-math-20k \
  --micro_train_batch_size 1 \
  --train_batch_size 24 \
  --micro_rollout_batch_size 1 \
  --rollout_batch_size 12 \
  --n_samples_per_prompt 3 \
  --max_samples 200 \
  --max_epochs 1 \
  --prompt_max_len 1024 \
  --generate_max_len 12000 \
  --zero_stage 3 \
  --bf16 \
  --actor_learning_rate 1e-7 \
  --init_kl_coef 0.01 \
  --prompt_data data/maths_dataset_eleuther.json \
  --input_key input \
  --apply_chat_template \
  --normalize_reward \
  --packing_samples \
  --flash_attn \
  --gradient_checkpointing \
  --use_wandb b936d9b7656895a60f4f4f993e3b1a7639b3430c \
  --wandb_project math_rl_full_length_sweep \
  --wandb_run_name am-test \
  --advantage_estimator grpo \
  --env_file MATH_rl_env \
  --env_class MathEnv \
  --adam_offload \
  #--colocate_actor_ref \
  #"TORCH_EXTENSIONS_DIR": "/data1/joey/torch_extensions", 
  
  # --lora_rank 16 \
  # --lora_alpha 32 \
  # --lora_dropout 0.05 \
#meta-llama/Meta-Llama-3-8B-Instruct