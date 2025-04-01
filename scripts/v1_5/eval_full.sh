#!/bin/bash
MODEL_PATH=$1

# CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mmvet.sh $MODEL_PATH
# CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/sqa.sh $MODEL_PATH
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash scripts/v1_5/eval/mmbench.sh $MODEL_PATH
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash scripts/v1_5/eval/textvqa.sh $MODEL_PATH
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/gqa.sh $MODEL_PATH
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash scripts/v1_5/eval/vizwiz.sh $MODEL_PATH


