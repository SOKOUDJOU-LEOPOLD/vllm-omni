import json

from huggingface_hub import hf_hub_download

REPO_ID = "HiDream-ai/HiDream-O1-Image"

config_path = hf_hub_download(REPO_ID, "config.json")
with open(config_path) as f:
    config = json.load(f)

print("Top-level config.json keys:", list(config.keys()))
print("architectures:", config.get("architectures"))
print("model_type:", config.get("model_type"))
print("_class_name:", config.get("_class_name"))
