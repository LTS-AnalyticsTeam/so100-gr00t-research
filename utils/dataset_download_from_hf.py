#!/usr/bin/env python
"""
Hugging Face Hub から複数 Dataset を条件付きでダウンロードするユーティリティ
Usage:
  python downloading_data.py --org lt-s \
      --tags move red_block \
      --name ".*train.*" \
      --dest-dir ./datasets

  # Collection 優先
  python downloading_data.py \
      --collection "TheBloke/recent-models-64f9a55bb3115b4f513ec026" \
      --dest-dir ./datasets
"""
import argparse, os, re, shutil, sys
from pathlib import Path
from typing import List, Set

from huggingface_hub import HfApi, snapshot_download

def normalize_tags(tags: List[str]) -> Set[str]:
    return set(t.lower() for t in tags) if tags else set()

def match_tags(info_tags: List[str], required: Set[str]) -> bool:
    return required.issubset(set(t.lower() for t in (info_tags or [])))

def main() -> None:
    parser = argparse.ArgumentParser(description="Download HF datasets with filters")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--org", help="Organization or user namespace")
    group.add_argument("--collection", help="Collection slug (owner/slug_id)")
    parser.add_argument("--tags", nargs="*", help="Filter by tag(s)")
    parser.add_argument("--name", help="Regex pattern on repo_id or dataset_name")
    parser.add_argument("--dest-dir", required=True, help="Local root directory")
    parser.add_argument("--token", help="HF token (env var HUGGING_FACE_HUB_TOKEN でも可)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel download workers")
    args = parser.parse_args()

    api = HfApi(token=args.token)

    # 1️⃣ 対象 Dataset repo_id を列挙
    repo_ids: List[str] = []
    if args.collection:
        coll = api.get_collection(args.collection)   # 取得  (要求が Collection の場合) 
        repo_ids = [
            item.item_id for item in coll.items
            if item.item_type == "dataset"
        ]
        if args.org:
            repo_ids = [rid for rid in repo_ids if rid.split("/")[0] == args.org]
    else:
        infos = api.list_datasets(author=args.org, full=True)
        tag_req = normalize_tags(args.tags)
        name_pat = re.compile(args.name) if args.name else None
        for info in infos:
            if tag_req and not match_tags(info.tags, tag_req):
                continue
            if name_pat and not name_pat.search(info.id):
                continue
            repo_ids.append(info.id)

    if not repo_ids:
        sys.exit("No dataset matched the given criteria.")

    # 2️⃣ ダウンロード
    dest_root = Path(args.dest_dir).expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    for rid in repo_ids:
        print(f"↓ {rid}")
        local_dir = dest_root / rid.replace("/", "__")
        # 既に存在する場合は上書きしたくなければスキップ
        snapshot_download(
            repo_id=rid,
            repo_type="dataset",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            token=args.token,
            max_workers=args.workers,
        )

if __name__ == "__main__":
    main()
