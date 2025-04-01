#!/bin/bash

MODEL_PATH=${1:-"liuhaotian/llava-v1.5-13b"}

python -m llava.eval.model_vqa \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/$(basename $MODEL_PATH).jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$(basename $MODEL_PATH).jsonl \
    --dst ./playground/data/eval/mm-vet/results/$(basename $MODEL_PATH).json

