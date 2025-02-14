uv run ray stop && 
uv lock --upgrade-package openrlhf && 
VLLM_CONFIGURE_LOGGING=0 uv run ray start --head --port 6380 --dashboard-port 8265 --num-gpus 8 --num-cpus 128 --dashboard-agent-listen-port 52366 --ray-client-server-port 9999 && 
./openrlhf_ppo.sh