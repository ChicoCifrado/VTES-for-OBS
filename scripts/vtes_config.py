"""
VTES card sets from static.krcg.org/card/set/ — mirrors mtg_card_detector/config.py.
"""
import os
from pathlib import Path

KRCG_CDN = "https://static.krcg.org/card"

# All known VTES sets (each is a directory under /card/set/)
ALL_SETS = [
    "anarch-unbound",
    "anarchs",
    "anarchs-and-alastor-storyline",
    "ancient-hearts",
    "anthology",
    "anthology-1",
    "anthology-reprint",
    "black-hand",
    "blood-shadowed-court",
    "bloodlines",
    "camarilla-edition",
    "cultists-storyline",
    "danse-macabre",
    "dark-sovereigns",
    "demo-decks",
    "ebony-kingdom",
    "fall-edens-legacy-storyline",
    "fall-of-london",
    "fifth-edition",
    "fifth-edition-anarch",
    "fifth-edition-companion",
    "final-nights",
    "first-blood",
    "full-bleed-promo",
    "gehenna",
    "heirs-to-the-blood",
    "heirs-to-the-blood-reprint",
    "humble-bundle",
    "infernal-storyline",
    "jyhad",
    "keepers-of-tradition",
    "keepers-of-tradition-reprint",
    "kickstarter-promo",
    "kindred-most-wanted",
    "legacies-of-blood",
    "lords-of-the-night",
    "lost-kindred",
    "new-blood",
    "new-blood-ii",
    "new-blood-iii",
    "nights-of-reckoning",
    "print-on-demand",
    "promo",
    "promo-pack-1",
    "promo-pack-2",
    "promo-pack-3",
    "sabbat",
    "sabbat-preconstructed",
    "sabbat-v5",
    "sabbat-war",
    "sword-of-caine",
    "tenth-anniversary",
    "the-unaligned",
    "third-edition",
    "thirtieth-anniversary",
    "twenty-fifth-anniversary",
    "twilight-rebellion",
    "vampire-the-eternal-struggle",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CARD_CACHE_DIR = DATA_DIR / "card-cache"
CARD_HASH_INDEX = DATA_DIR / "card-hash-index.json"
