import sys
import os
import torch

# 1. 设置路径 (请确保这些是服务器上的绝对路径)
# Llama-Factory 源码路径
LF_SRC = "/llm-reco-ssd-share/baohonghui/LlamaFactory/src"
sys.path.append(LF_SRC)

from llamafactory.hparams import ModelArguments
from llamafactory.model.loader import load_model, load_tokenizer

# 架构路径 (包含你修改后的 modeling_qwen3.py 和 config.json)
model_arch_path = "/llm-reco-ssd-share/baohonghui/LlamaFactory/model/qwen3"
# 权重路径 (Checkpoints)
remote_weights = "/llm-reco-ssd-share/zhangzixing/onerec_pretrain/model_output/pro/sft/8b_v0.1.0_fromstg2_noamazon/step6500/global_step6500/converted"

def main():
    print(f"--- 正在加载自定义模型 ---")
    print(f"架构路径: {model_arch_path}")
    print(f"权重路径: {remote_weights}")

    # 2. 构造参数
    # 使用我们之前修改过的 ModelArguments，它支持 model_weight_path 
    model_args = ModelArguments(
        model_name_or_path=model_arch_path,
        model_weight_path=remote_weights,
        trust_remote_code=True
    )

    # 3. 使用 Llama-Factory 的原生 loader 加载
    # 它会自动处理“软链接缝合”、词表扩容、架构对齐等所有逻辑
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    
    model = load_model(tokenizer, model_args, finetuning_args=None, is_trainable=False)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    print(f"\n--- 模型加载成功 ---")
    
    # 4. 测试生成
    prompt = "User: 你现在是一个推荐系统助手，请给出建议。\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print(f"\n测试输入: {prompt}")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"\n模型输出: \n{response}")

if __name__ == "__main__":
    main()
