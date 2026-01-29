#!/bin/bash

# 设置 Python 路径
export PYTHONPATH=$PYTHONPATH:/llm-reco-ssd-share/baohonghui/LlamaFactory/src

# 运行推理测试
# 建议指定一张卡即可，推理不需要多卡
CUDA_VISIBLE_DEVICES=0 python /llm-reco-ssd-share/baohonghui/LlamaFactory/honghui_infer_custom.py
