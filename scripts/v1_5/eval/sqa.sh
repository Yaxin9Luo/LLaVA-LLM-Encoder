#!/bin/bash

MODEL_PATH=${1:-"liuhaotian/llava-v1.5-13b"}

python -m llava.eval.model_vqa_science \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/$(basename $MODEL_PATH).jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$(basename $MODEL_PATH).jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/$(basename $MODEL_PATH)_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/$(basename $MODEL_PATH)_result.json
