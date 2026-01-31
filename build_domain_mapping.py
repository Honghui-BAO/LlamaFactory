import json
import os

def build_domain_mapping(datamaps_path):
    """
    Builds a mapping from internal numeric user_id (string) to a domain_id (0-9).
    Based on the format: "DomainName::OriginalUserId": "InternalId"
    """
    if not os.path.exists(datamaps_path):
        print(f"Warning: {datamaps_path} not found.")
        return {}

    with open(datamaps_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    user2id = data.get('user2id', {})
    
    # 1. Extract unique domains
    # Keys look like "Home_and_Kitchen::A1ZEPNF3VWAZU7"
    unique_domains = sorted(list(set([k.split('::')[0] for k in user2id.keys() if '::' in k])))
    
    # 2. Map domain name to index (0-9)
    # The user mentioned 10 unique domains. We filter/limit if necessary.
    domain2idx = {domain: i for i, domain in enumerate(unique_domains[:10])}
    
    print(f"Found {len(unique_domains)} unique domains. Mapped first 10 to indices:")
    for d, idx in domain2idx.items():
        print(f"  {d} -> {idx}")

    # 3. Create final user_id -> domain_id mapping
    user_id2domain_id = {}
    for key, internal_id in user2id.items():
        if '::' in key:
            domain_name = key.split('::')[0]
            if domain_name in domain2idx:
                user_id2domain_id[str(internal_id)] = domain2idx[domain_name]
    
    return user_id2domain_id

if __name__ == "__main__":
    # Example usage
    PATH = "/llm-reco-ssd-share/baohonghui/Reference/word_identifier/raw_data/merged_datamaps.json"
    mapping = build_domain_mapping(PATH)
    print(f"Built mapping for {len(mapping)} users.")
    
    # Save for training script usage
    OUTPUT_PATH = "/llm-reco-ssd-share/baohonghui/LlamaFactory/user_id2domain_id.json"
    with open(OUTPUT_PATH, "w") as f:
        json.dump(mapping, f)
    print(f"Saved to {OUTPUT_PATH}")
