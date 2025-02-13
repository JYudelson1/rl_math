uv run ray stop && 
uv lock --upgrade-package openrlhf && 
VLLM_CONFIGURE_LOGGING=0 CUDA_VISIBLE_DEVICES=0,1,3,4,5,6 uv run ray start --head --port 6380 --dashboard-port 8265 --num-gpus 6 --dashboard-agent-listen-port 52366 --ray-client-server-port 9999 --temp-dir /data1/joey/tmp && 
./openrlhf_ppo.sh