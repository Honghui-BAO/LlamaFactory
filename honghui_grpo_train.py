import re
import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
import transformers

# --- Configuration & Paths ---
MODEL_PATH = os.getenv("MODEL_PATH", "/llm-reco-ssd-share/baohonghui/ckpt/converted")
DATA_PATH = os.getenv("DATA_PATH", "/llm-reco-ssd-share/baohonghui/LlamaFactory/data/sum_rec/amazon_merged_sft_data_rec_honghui_v2.json")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/llm-reco-ssd-share/baohonghui/torchrec/SumRec/merge/ckpt/reasoner_grpo_v1")

# --- Custom Reward Functions ---

def format_reward_func(completions, **kwargs) -> list[float]:
    """
    Rewards completions that follow the specific <|sid_begin|>...<|sid_end|> format.
    """
    pattern = r"^Item: <\|sid_begin\|>.*?<\|sid_end\|>"
    rewards = []
    for content in completions:
        # Check if the generated content starts with the required Item format
        if re.search(pattern, content):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def accuracy_reward_func(completions, test_ground_truth_identifier, **kwargs) -> list[float]:
    """
    Rewards completions where the generated identifier matches the ground truth identifier.
    """
    rewards = []
    for content, gt_id in zip(completions, test_ground_truth_identifier):
        # Extract the part between <|sid_begin|> and <|sid_end|>
        match = re.search(r"<\|sid_begin\|>(.*?)<\|sid_end\|>", content)
        if match:
            generated_id = f"<|sid_begin|>{match.group(1)}<|sid_end|>"
            if generated_id.strip() == gt_id.strip():
                rewards.append(1.0)
            else:
                # Partial/Fuzzy Reward could be added here
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

# --- Domain Mapping ---
DOMAIN_MAPPING_V2_PATH = "/llm-reco-ssd-share/baohonghui/LlamaFactory/user_id2domain_id.json"
user_id2domain_id = {}
if os.path.exists(DOMAIN_MAPPING_V2_PATH):
    with open(DOMAIN_MAPPING_V2_PATH, "r") as f:
        user_id2domain_id = json.load(f)
    print(f"Loaded domain mapping for {len(user_id2domain_id)} users.")
else:
    print(f"Warning: {DOMAIN_MAPPING_V2_PATH} not found. MoE Reasoner will use Shared Expert only.")

# --- Data Preparation ---

def prepare_dataset(path):
    dataset = load_dataset("json", data_files=path, split="train")
    
    def map_fn(examples):
        prompts = []
        domain_ids_list = []
        
        for i in range(len(examples["instruction"])):
            instr = examples["instruction"][i]
            inp = examples["input"][i]
            metadata = examples["metadata"][i]
            
            # Construct standard prompt
            prompts.append(f"{instr}\n{inp}")
            
            # Lookup domain_id dynamically
            u_id = str(metadata.get("user_id", ""))
            domain_ids_list.append(user_id2domain_id.get(u_id, -1)) # -1 defaults to shared only
            
        return {
            "prompt": prompts,
            "domain_ids": domain_ids_list,
            "test_ground_truth_identifier": examples["test_ground_truth_identifier"]
        }
    
    return dataset.map(map_fn, batched=True, remove_columns=dataset.column_names)

# --- Training Script ---

def main():
    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # Crucial for generation

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None # Managed by Accelerator
    )

    # 2. Freeze Backbone, Train Reasoner
    # We only want to update the GatedMLP Reasoner
    for name, param in model.named_parameters():
        if "reasoner" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--- Trainable Parameters: {trainable_params} ---")

    # 3. Load Dataset
    dataset = prepare_dataset(DATA_PATH)

    # 4. GRPO Configuration
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_generations=8,     # Group size G
        max_prompt_length=512,
        max_completion_length=256,
        num_train_epochs=1,
        save_steps=100,
        logging_steps=1,
        bf16=True,
        report_to="none",
        warmup_steps=50,
        remove_unused_columns=False, # Keep ground truth for rewards
        # GRPO specific hyperparams
        beta=0.01, # KL penalty
    )

    # 5. Initialize Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward_func, accuracy_reward_func],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # 6. Train!
    print("--- Starting GRPO Training ---")
    trainer.train()
    
    # 7. Save
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"--- Training Complete. Model saved to {OUTPUT_DIR} ---")

if __name__ == "__main__":
    main()
