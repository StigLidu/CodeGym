# Copyright (c) 2025 CodeGym Project Team.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc/4.0/

import os
from requests.exceptions import HTTPError
import sys
from pathlib import Path
from typing import Optional


def hf_download(
    repo_id: Optional[str] = None,
    hf_token: Optional[str] = None,
    local_dir: Optional[str] = None,
    upload_hdfs_path: Optional[str] = None, 
) -> None:
    from huggingface_hub import snapshot_download

    local_dir = local_dir or "checkpoints"

    os.makedirs(f"{local_dir}/{repo_id}", exist_ok=True)
    try:
        snapshot_download(
            repo_id,
            local_dir=f"{local_dir}/{repo_id}",
            local_dir_use_symlinks=False,
            token=hf_token,
            force_download=True,
            max_workers=1,
            etag_timeout=30
        )
    except HTTPError as e:
        print(e.response)
        if e.response.status_code == 401:
            print(
                "You need to pass a valid `--hf_token=...` to download private checkpoints."
            )
        else:
            raise e

    if upload_hdfs_path:
        print(f"Uploading to {upload_hdfs_path}")
        os.system(f"hdfs dfs -put -f {local_dir}/{repo_id} {upload_hdfs_path}")
        print(f"Uploaded to {upload_hdfs_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download data from HuggingFace Hub.")
    parser.add_argument(
        "--repo_id",
        type=str,
        help="Repository ID to download from.",
    )
    parser.add_argument(
        "--local_dir", type=str, default=None, help="Local directory to download to."
    )
    parser.add_argument(
        "--hf_token", type=str, default=None, help="HuggingFace API token."
    )
    parser.add_argument("--upload_hdfs_path", type=str, default=None, help="HDFS path to upload to.")
    args = parser.parse_args()
    hf_download(args.repo_id, args.hf_token, args.local_dir, args.upload_hdfs_path)
