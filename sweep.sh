export LEARNING_RATE=5e-8
export KL_COEF=0.01
export RUN_NAME=Qwen32B—Distill_5e-8lr_0-01kl

./full_runner.sh

export LEARNING_RATE=1e-7
export KL_COEF=0.01
export RUN_NAME=Qwen32B—Distill_1e-7lr_0-01kl

./full_runner.sh &&

export LEARNING_RATE=3e-7
export KL_COEF=0.01
export RUN_NAME=Qwen32B—Distill_3e-7lr_0-01kl

./full_runner.sh &&

export LEARNING_RATE=7e-7
export KL_COEF=0.01
export RUN_NAME=Qwen32B—Distill_7e-7lr_0-01kl

./full_runner.sh &&

export LEARNING_RATE=1e-6
export KL_COEF=0.01
export RUN_NAME=Qwen32B—Distill_1e-6lr_0-01kl

./full_runner.sh

