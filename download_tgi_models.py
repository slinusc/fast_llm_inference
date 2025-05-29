from huggingface_hub import snapshot_download

for repo_id in [
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b-it"
]:
    print(f"Downloading {repo_id} ...")
    snapshot_download(
    repo_id=repo_id,
    cache_dir="/home/ubuntu/tgi_cache",
    local_dir_use_symlinks=False,
    force_download=True,
    )

    print(f"✓ Done: {repo_id}")
