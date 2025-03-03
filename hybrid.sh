set -x
source .env

uv lock --upgrade-package openrlhf

uv run ray stop > /dev/null 2>&1 #Suppress the output of ray stop

CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 uv run ray start --head --port 6380 --num-gpus 7 > /dev/null 2>&1 #--num-cpus 128
uv run ray job submit --address="http://127.0.0.1:8265" \
  --working-dir . \
  --runtime-env-json='{
  "setup_commands": ["pip install openrlhf[vllm]"], 
  "env_vars": {}}' \
  -- python -m openrlhf.cli.train_ppo_ray \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 6 \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 6 \
  --vllm_num_engines 2 \
  --vllm_tensor_parallel_size 1 \
  --pretrain deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --save_path /data1/joey/rl_math/checkpoint/test-rl-math-20k \
  --micro_train_batch_size 1 \
  --train_batch_size 16 \
  --micro_rollout_batch_size 4 \
  --rollout_batch_size 16 \
  --n_samples_per_prompt 8 \
  --max_samples 200 \
  --max_epochs 1 \
  --prompt_max_len 4000 \
  --generate_max_len 8000 \
  --zero_stage 3 \
  --bf16 \
  --actor_learning_rate 1e-7 \
  --init_kl_coef 0.01 \
  --use_kl_loss \
  --use_kl_estimator_k3 \
  --prompt_data data/maths_dataset_eleuther.json \
  --input_key input \
  --apply_chat_template \
  --normalize_reward \
  --packing_samples \
  --flash_attn \
  --gradient_checkpointing \
  --use_wandb b936d9b7656895a60f4f4f993e3b1a7639b3430c \
  --wandb_project math_rl_full_length_sweep \
  --wandb_run_name math-env-test-14b \
  --advantage_estimator grpo \
  --env_file MATH_rl_env \
  --env_class MathEnv \
  --adam_offload \
  --eval_steps 1 \
  --enforce_eager
  #--colocate_actor_ref \
  #"TORCH_EXTENSIONS_DIR": "/data1/joey/torch_extensions", 
  
  # --lora_rank 16 \
  # --lora_alpha 32 \
  # --lora_dropout 0.05 \
#meta-llama/Meta-Llama-3-8B-Instruct
