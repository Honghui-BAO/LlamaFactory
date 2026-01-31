import json
import os
import argparse

def patch_file(file_path, mapping_path, output_path=None):
    if not os.path.exists(file_path):
        print(f"Error: Dataset {file_path} not found.")
        return
    if not os.path.exists(mapping_path):
        print(f"Error: Mapping {mapping_path} not found.")
        return

    print(f"Loading mapping from {mapping_path}...")
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)

    print(f"Loading dataset from {file_path}...")
    with open(file_path, 'r') as f:
        dataset = json.load(f)

    patched_count = 0
    missing_count = 0
    
    for item in dataset:
        # Extract user_id from metadata
        user_id = item.get("metadata", {}).get("user_id")
        if user_id:
            # Map to domain_id
            domain_id = mapping.get(str(user_id))
            if domain_id is not None:
                item["domain_ids"] = domain_id  # Named domain_ids to match model forward arg
                patched_count += 1
            else:
                item["domain_ids"] = -1 # Default to shared expert
                missing_count += 1
        else:
            item["domain_ids"] = -1
            missing_count += 1

    if output_path is None:
        output_path = file_path # In-place patch by default or choose new name

    print(f"Saving patched dataset to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Finish! Patched: {patched_count}, Defaulted (-1): {missing_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch LlamaFactory SFT dataset with domain_ids for MoE Reasoner.")
    parser.add_argument("--file", type=str, required=True, help="Path to the SFT JSON file.")
    parser.add_argument("--mapping", type=str, default="/llm-reco-ssd-share/baohonghui/LlamaFactory/user_id2domain_id.json", help="Path to user_id2domain_id.json.")
    parser.add_argument("--output", type=str, help="Path to save the patched file (defaults to in-place).")
    
    args = parser.parse_args()
    patch_file(args.file, args.mapping, args.output)
