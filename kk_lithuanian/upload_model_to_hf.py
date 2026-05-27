import argparse
import os
from huggingface_hub import HfApi

HF_USERNAME = "Zygelis"
DEFAULT_BRANCH = "main"
DEFAULT_MODEL_REPO = "qwen3-1.7B-lt-dapo-v1"
DEFAULT_LOCAL_MODEL_DIR = "kk_lithuanian/trained_models/qwen3-1.7B-lt-dapo-step-312-merged-lora"
PRIVATE_REPO = True

def ensure_branch_exists(api, repo_id, branch, token, repo_type="model"):
    """Create a branch if it doesn't exist."""
    try:
        api.create_branch(repo_id=repo_id, branch=branch, repo_type=repo_type, token=token)
        print(f"Created branch '{branch}'")
    except Exception as e:
        if "already exists" in str(e):
            print(f"Branch '{branch}' already exists")
        else:
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a merged model folder to Hugging Face")
    parser.add_argument("--branch", default=DEFAULT_BRANCH, help="Branch/revision to upload to (default: main)")
    parser.add_argument("--repo-name", default=DEFAULT_MODEL_REPO, help="Model repo name on Hugging Face")
    parser.add_argument(
        "--local-model-dir",
        default=DEFAULT_LOCAL_MODEL_DIR,
        help="Local directory to upload",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=PRIVATE_REPO,
        help="Create the model repo as private",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Export it before running.")

    api = HfApi()
    model_repo_id = f"{HF_USERNAME}/{args.repo_name}"
    api.create_repo(repo_id=model_repo_id, private=args.private, exist_ok=True, token=token)
    ensure_branch_exists(api, model_repo_id, args.branch, token, repo_type="model")
    api.upload_folder(
        folder_path=args.local_model_dir,
        repo_id=model_repo_id,
        repo_type="model",
        token=token,
        revision=args.branch,
    )
    print(f"Uploaded model to https://huggingface.co/{model_repo_id}/tree/{args.branch}")


if __name__ == "__main__":
    main()
