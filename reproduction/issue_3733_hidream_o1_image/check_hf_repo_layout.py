from huggingface_hub import HfApi

REPO_ID = "HiDream-ai/HiDream-O1-Image"

api = HfApi()
files = api.list_repo_files(REPO_ID)

print(f"Files in {REPO_ID}:")
for f in sorted(files):
    print(" -", f)

print()
print("Has model_index.json:", "model_index.json" in files)
print("Has model.safetensors.index.json:", "model.safetensors.index.json" in files)
