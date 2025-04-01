#!/bin/bash

MODEL_PATH=${1:-"liuhaotian/llava-v1.5-13b"}
MODEL_NAME=$(basename $MODEL_PATH)
SPLIT="mmbench_dev_en_20231003"

# python -m llava.eval.model_vqa_mmbench \
#     --model-path $MODEL_PATH \
#     --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
#     --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$MODEL_NAME.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $MODEL_NAME
