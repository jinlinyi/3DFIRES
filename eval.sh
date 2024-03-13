#!/bin/bash
echo "EVALUATE on 3 views"
INPUT_VIEW=3
python eval.py \
    --ckpt-path="ckpts/model_0299999.pth" \
    --cfg-path="ckpts/config.yaml" \
    --output-dir="output" \
    --dataset-name="test_set$INPUT_VIEW" \
    --test-view=$INPUT_VIEW


echo "EVALUATE on 5 views"
INPUT_VIEW=5
python eval.py \
    --ckpt-path="ckpts/model_0299999.pth" \
    --cfg-path="ckpts/config.yaml" \
    --output-dir="output" \
    --dataset-name="test_set$INPUT_VIEW" \
    --test-view=$INPUT_VIEW
    