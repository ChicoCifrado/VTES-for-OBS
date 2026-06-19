#!/usr/bin/env python3
"""
Organize downloaded card cache images into dataset/by_type/{Type}/ structure.

Usage:
    python3 scripts/organize_dataset.py \
        --vtes-json /mnt/d/Hermes/vtes/vtes.json \
        --cache-dir /mnt/d/Hermes/vtes/data/card-cache \
        --output-dir /mnt/d/Hermes/vtes/dataset/by_type
"""

import argparse
import json
import re
import shutil
from pathlib import Path


VTES_TYPES = [
    "Action", "Action Modifier", "Ally", "Combat", "Conviction",
    "Equipment", "Event", "Imbued", "Master", "Political Action",
    "Power", "Reaction", "Retainer", "Vampire",
]


def norm(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', name.lower())


def main():
    parser = argparse.ArgumentParser(description="Organize card cache by type")
    parser.add_argument("--vtes-json", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    with open(args.vtes_json, encoding="utf-8") as f:
        cards = json.load(f)

    # Build card name → primary type mapping
    card_type_map = {}
    for card in cards:
        types = card.get("types", [])
        if not types:
            continue
        primary = types[0]
        names = set()
        for field in ["printed_name", "name", "_name"]:
            if field in card and card[field]:
                names.add(card[field])
        for variant in card.get("name_variants", []):
            names.add(variant)
        for name in names:
            n = norm(name)
            if n and n not in card_type_map:
                card_type_map[n] = primary

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_dir():
        print(f"ERROR: Cache dir not found: {cache_dir}")
        print("Run python3 scripts/fetch_vtes_data.py first")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create type subdirectories
    for t in VTES_TYPES:
        (out_dir / t).mkdir(exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    copied = 0
    unmatched = 0

    for img_path in sorted(cache_dir.iterdir()):
        if img_path.suffix.lower() not in exts:
            continue
        stem = img_path.stem
        n = norm(stem)
        card_type = card_type_map.get(n)
        if card_type is None:
            unmatched += 1
            continue
        if card_type not in VTES_TYPES:
            continue
        dst = out_dir / card_type / img_path.name
        shutil.copy2(img_path, dst)
        copied += 1

    # Print stats
    print(f"Organized {copied} images by type, {unmatched} unmatched")
    for t in VTES_TYPES:
        count = len(list((out_dir / t).iterdir()))
        print(f"  {t}: {count}")


if __name__ == "__main__":
    main()
