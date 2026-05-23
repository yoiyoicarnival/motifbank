#!/usr/bin/env python3
"""
HF Spaces デプロイスクリプト
Usage: python3 deploy_hf.py --token YOUR_HF_TOKEN --username YOUR_HF_USERNAME
"""
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('--token',    required=True, help='HuggingFace API token')
parser.add_argument('--username', required=True, help='HuggingFace username')
parser.add_argument('--name',     default='ai-credibility-checker')
args = parser.parse_args()

repo_id = f"{args.username}/{args.name}"
space_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hf_space')

from huggingface_hub import HfApi, login

login(token=args.token)
api = HfApi()

# Space 作成 (既存なら無視)
try:
    api.create_repo(
        repo_id=repo_id,
        repo_type='space',
        space_sdk='streamlit',
        exist_ok=True,
        private=False,
    )
    print(f"Space created: https://huggingface.co/spaces/{repo_id}")
except Exception as e:
    print(f"create_repo: {e}")

# ファイル一括アップロード
print(f"Uploading from {space_dir} ...")
api.upload_folder(
    folder_path=space_dir,
    repo_id=repo_id,
    repo_type='space',
)
print(f"\n✅ Done! → https://huggingface.co/spaces/{repo_id}")
print("  ※ 初回ビルドに3〜5分かかります (GPT-2 ダウンロード)")
