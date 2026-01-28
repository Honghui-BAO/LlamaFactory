# 核心逻辑：解耦 架构路径 和 权重路径
# 1. 架构代码路径（包含你修改过的 modeling_qwen3.py 和 config.json）
# 注意：在服务器上启动时，请确保此路径是服务器上的绝对路径
model_arch_path="/Users/honghuibao/Desktop/Baselines/LlamaFactory/model/qwen3"

# 2. 远程权重路径（只读参数）
remote_weights="/llm-reco-ssd-share/zhangzixing/onerec_pretrain/model_output/pro/sft/8b_v0.1.0_fromstg2_noamazon/step6500/global_step6500/converted"

output_dir=/llm-reco-ssd-share/baohonghui/Reference/torchrec/SumRec/merge/ckpt/rmdr_v15_0_single_avg_custom
export WANDB_DISABLED=true

# 我们现在使用新添加的 --model_weight_path 参数
# --model_name_or_path: 指向包含自定义代码和 config 的文件夹
# --model_weight_path: 指向原始权重文件夹

nohup deepspeed --num_gpus 8 \
src/train.py \
--deepspeed examples/deepspeed/ds_z2_config.json \
--stage sft \
--model_name_or_path $model_arch_path \
--model_weight_path $remote_weights \
--trust_remote_code True \
--do_train \
--dataset meta2sid_dataset,rec_dataset \
--use_reasoner \
--reasoner_layers 1 \
--template qwen3 \
--finetuning_type full \
--output_dir  $output_dir \
--overwrite_cache \
--save_total_limit 3 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--save_steps 200 \
--learning_rate 1e-4 \
--num_train_epochs 3.0 \
--plot_loss \
--bf16 > train_custom_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "Training started using separate Arch and Weight paths. Log: train_custom_*.log"
