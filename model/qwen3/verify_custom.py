import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

model_path = "/Users/honghuibao/Desktop/Baselines/LlamaFactory/model/qwen3"

print("--- Step 1: Loading Config ---")
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
print(f"Loaded architecture: {config.architectures}")
print(f"use_custom_logic flag: {getattr(config, 'use_custom_logic', False)}")

print("\n--- Step 2: Loading Model (this might take a while if model is large) ---")
# Use device_map="auto" or specific device if needed. 
# For small verification, we can use torch_dtype=torch.float16 or bfloat16
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True,
    device_map="cpu", # Change to "auto" or "cuda" if you have GPU
    torch_dtype=torch.bfloat16
)

print(f"\nModel Class: {model.__class__.__name__}")

print("\n--- Step 3: Running Dummy Forward Pass ---")
input_ids = torch.tensor([[1, 2, 3]]) # Just a dummy input
with torch.no_grad():
    outputs = model(input_ids)

print("\n--- Step 4: Verification Result ---")
if "DEBUG: Applying custom logic to hidden_states" in str(outputs):
    # Note: Output won't contain the print, but it should have printed to stdout during forward
    pass

print("\nDone. Check the console output for 'DEBUG: Applying custom logic to hidden_states'.")
