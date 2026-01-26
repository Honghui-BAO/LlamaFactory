model_path=/llm-reco-ssd-share/zhangzixing/onerec_pretrain/model_output/pro/sft/8b_v0.1.0_fromstg2_noamazon/step6500/global_step6500/converted
output_dir=/llm-reco-ssd-share/baohonghui/Reference/torchrec/SumRec/merge/ckpt/rmdr_v15_0_single_avg
export WANDB_DISABLED=true

nohup deepspeed --num_gpus 8 \
src/train.py \
--deepspeed examples/deepspeed/ds_z2_config.json \
--stage sft \
--model_name_or_path $model_path \
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
--bf16 > train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "Training started in background. Log file: train_$(date +%Y%m%d_%H%M%S).log"
echo "Use 'tail -f train_*.log' to monitor progress"