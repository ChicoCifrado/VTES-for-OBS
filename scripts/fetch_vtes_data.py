"""
Fetch VTES card images from static.krcg.org.
Mirrors mtg_card_detector/fetch_data.py — uses Scryfall-style API via krcg.org.

Usage:
  python fetch_vtes_data.py              # download all missing cards
  python fetch_vtes_data.py --update-index  # rebuild card-hash-index.json
"""
import json
import os
import sys
import hashlib
import urllib.request
from pathlib import Path

from vtes_config import KRCG_CDN, CARD_CACHE_DIR, CARD_HASH_INDEX


def get_valid_filename(s):
    """Sanitize a string for use as a filename."""
    s = str(s).strip().replace(" ", "_").replace("'", "").replace('"', "")
    return "".join(c for c in s if c.isalnum() or c in "_-.")


def fetch_card_image(card_url, cache_path):
    """Download a card image from krcg.org to cache."""
    if cache_path.exists():
        return True
    try:
        urllib.request.urlretrieve(card_url, filename=str(cache_path))
        return True
    except Exception as e:
        print(f"  WARNING: Failed to download {card_url}: {e}")
        return False


def download_all_cards(vtes_json_path, max_cards=None):
    """Download all card images listed in vtes.json."""
    with open(vtes_json_path, "r", encoding="utf-8") as f:
        cards = json.load(f)

    os.makedirs(CARD_CACHE_DIR, exist_ok=True)

    count = 0
    for card in cards:
        card_id = card.get("id", "")
        card_name = card.get("name", "")
        card_url = card.get("url", "")

        if not card_url:
            continue

        safe_name = get_valid_filename(card_name)
        cache_path = CARD_CACHE_DIR / f"{safe_name}.jpg"

        if fetch_card_image(card_url, cache_path):
            count += 1
            if count % 500 == 0:
                print(f"  Downloaded {count} cards")

        if max_cards and count >= max_cards:
            break

    print(f"Downloaded/verified {count} card images")
    return count


def compute_average_hash(image_path):
    """Compute 64-bit average hash (like card-matcher.js)."""
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print("Pillow required: pip install Pillow")
        return None

    img = Image.open(image_path).convert("L").resize((8, 8), Image.LANCZOS)
    pixels = list(img.getdata())
    mean = sum(pixels) / len(pixels)
    h = 0
    for i, p in enumerate(pixels):
        if p > mean:
            h |= 1 << (63 - i)
    return h


def main():
    vtes_json_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "vtes.json"
    )

    if not os.path.exists(vtes_json_path):
        print(f"ERROR: vtes.json not found at {vtes_json_path}")
        sys.exit(1)

    print(f"Downloading cards from krcg.org...")
    print(f"  Source: {vtes_json_path}")
    print(f"  Cache: {CARD_CACHE_DIR}")

    count = download_all_cards(vtes_json_path)
    print(f"  Total: {count} cards in cache")

    if "--update-index" in sys.argv:
        print("\nRebuilding card-hash-index.json...")
        # Load existing index
        if CARD_HASH_INDEX.exists():
            with open(CARD_HASH_INDEX, "r") as f:
                index = json.load(f)
        else:
            index = {"byId": {}, "byZoneHash": {}, "byHash": {}, "byName": {}}

        with open(vtes_json_path, "r") as f:
            cards = json.load(f)

        for card in cards:
            card_id = card.get("id", "")
            card_name = card.get("name", "")
            cache_path = CARD_CACHE_DIR / f"{get_valid_filename(card_name)}.jpg"

            if not cache_path.exists():
                continue

            h = compute_average_hash(str(cache_path))
            if h is not None:
                hash_hex = format(h, "016x")
                if card_id in index.get("byId", {}):
                    index["byId"][card_id]["card_hash"] = hash_hex
                if card_name not in index.get("byName", {}):
                    index["byName"][card_name] = hash_hex

        with open(CARD_HASH_INDEX, "w") as f:
            json.dump(index, f, indent=2)
        print("  Updated card-hash-index.json")


if __name__ == "__main__":
    main()
