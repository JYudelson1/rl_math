# WITH TMP ACCESS, REMOVE MOST OF THESE
export TMPDIR=/data1/joey/tmp
export TMP=/data1/joey/tmp
export TEMP=/data1/joey/tmp
export TEMPDIR=/data1/joey/tmp
export GCC_TMPDIR=/data1/joey/tmp
export NVCC_TMPDIR=/data1/joey/tmp
export TORCH_EXTENSIONS_DIR=/data1/joey/torch_extensions
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
  --working-dir /data1/joey/rl_math \
  --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"], "env_vars": {"TMPDIR": "/data1/joey/tmp", "TMP": "/data1/joey/tmp", "TEMP": "/data1/joey/tmp", "TEMPDIR": "/data1/joey/tmp", "GCC_TMPDIR": "/data1/joey/tmp", "NVCC_TMPDIR": "/data1/joey/tmp", "TORCH_EXTENSIONS_DIR": "/data1/joey/torch_extensions", "HOME": "/data1/joey", "DS_BUILD_TEMP_DIR": "/data1/joey/tmp", "CCACHE_TEMPDIR": "/data1/joey/tmp", "HF_HOME": "/data1/joey/hf_cache", "RAY_BACKEND_LOG_LEVEL": "debug"}}' \
  -- python -m openrlhf.cli.train_ppo_ray \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 4 \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 4 \
  --vllm_num_engines 2 \
  --vllm_tensor_parallel_size 1 \
  --colocate_actor_ref \
  --pretrain deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --save_path /data1/joey/rl_math/checkpoint/test-rl-math-20k \
  --micro_train_batch_size 4 \
  --train_batch_size 64 \
  --micro_rollout_batch_size 3 \
  --rollout_batch_size 24 \
  --n_samples_per_prompt 3 \
  --max_samples 1000 \
  --max_epochs 1 \
  --prompt_max_len 1024 \
  --generate_max_len 20000 \
  --zero_stage 3 \
  --bf16 \
  --actor_learning_rate $LEARNING_RATE \
  --init_kl_coef $KL_COEF \
  --prompt_data data/maths_dataset_eleuther.json \
  --input_key input \
  --apply_chat_template \
  --normalize_reward \
  --packing_samples \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --use_wandb $WANDB_API_KEY \
  --wandb_project math_rl_full_length_sweep \
  --wandb_run_name $RUN_NAME \
  --advantage_estimator grpo \
  --remote_rm_url http://localhost:5000/get_reward \
  --vllm_sync_backend nccl \
  --eval_steps 1 \
  # --lora_rank 16 \
  # --lora_alpha 32 \
  # --lora_dropout 0.05 \
#meta-llama/Meta-Llama-3-8B-Instruct