## To start ray cluster
# VLLM_CONFIGURE_LOGGING=0 CUDA_VISIBLE_DEVICES=4,5,6 uv run ray start --head --port 6381 --dashboard-port 8266 --num-gpus 3 --dashboard-agent-listen-port 52366 --temp-dir /data1/joey/tmp

## To get ray status
# uv run ray status --address=0.0.0.0:6380

## To stop ray cluster
# uv run ray stop