#!/usr/bin/env python3
"""Process card photos: convert HEIC → cropped, rotated, properly named JPGs.

Reads from frontend/card_pictures/*.HEIC, outputs to frontend/public/cards/.
Detects card boundaries, crops out background, rotates to portrait, and
generates the mapping file for the frontend.
"""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageFilter
from pillow_heif import register_heif_opener

register_heif_opener()

SRC_DIR = Path("frontend/card_pictures")
OUT_DIR = Path("frontend/public/cards")
TARGET_HEIGHT = 500  # px, maintaining card aspect ratio (~1:1.8)
TARGET_WIDTH = int(TARGET_HEIGHT / 1.8)  # ~278

# Group photos (multiple cards on table) — skip these
GROUP_PHOTOS = {"IMG_1637", "IMG_1638", "IMG_1639", "IMG_1642"}

# ── Card identification ──────────────────────────────────────────────
# Mapping: IMG filename stem → (card_type, value, suit)
# Taroks: card_type="tarok", value=1-22, suit=None
# Suits:  card_type="suit", value=1-8 (SuitRank enum), suit="hearts"|"diamonds"|"clubs"|"spades"
#   value: 1=PIP_1, 2=PIP_2, 3=PIP_3, 4=PIP_4, 5=JACK, 6=KNIGHT, 7=QUEEN, 8=KING
#   PIP_1..4 for red suits = 1,2,3,4 pips; for black suits = 7,8,9,10 pips

CARD_MAP = {
    # ── Taroks ──
    "IMG_1656": ("tarok", 1, None),      # I (Pagat) — girl in fancy dress
    "IMG_1640": ("tarok", 2, None),      # II — eagle with "Industrie und Glück" banner
    "IMG_1672": ("tarok", 3, None),      # III — village farming scene
    "IMG_1679": ("tarok", 5, None),      # V — couple embracing by carriage
    "IMG_1649": ("tarok", 6, None),      # VI — women walking on mountain path
    "IMG_1658": ("tarok", 7, None),      # VII — couple at well/farmhouse
    "IMG_1669": ("tarok", 9, None),      # IX — couple in garden/field
    "IMG_1680": ("tarok", 11, None),     # XI — boy and woman at cistern
    "IMG_1652": ("tarok", 12, None),     # XII — man and maid in vineyard
    "IMG_1651": ("tarok", 15, None),     # XV — couple kissing by house
    "IMG_1645": ("tarok", 16, None),     # XVI — old woman, couple, village
    "IMG_1643": ("tarok", 17, None),     # XVII — boat scene on lake
    "IMG_1674": ("tarok", 18, None),     # XVIII — two women at balustrade
    "IMG_1641": ("tarok", 19, None),     # XIX — woman seated under tree
    "IMG_1655": ("tarok", 20, None),     # XX — pastoral couple with cow
    # Note: IV, VIII, X, XIII, XIV, XXI (Mond), XXII (Škis) not photographed

    # ── Hearts ♥ (red suit: pips show 1-4 symbols) ──
    "IMG_1675": ("suit", 1, "hearts"),   # 1♥ — single heart with Piatnik horse logo
    "IMG_1677": ("suit", 3, "hearts"),   # 3♥ — three hearts
    "IMG_1681": ("suit", 4, "hearts"),   # 4♥ — four hearts
    "IMG_1650": ("suit", 5, "hearts"),   # J♥ (Jack) — male figure with mustache, scepter
    "IMG_1654": ("suit", 7, "hearts"),   # Q♥ (Queen) — female figure with veil, pearls, flowers
    "IMG_1676": ("suit", 8, "hearts"),   # K♥ (King) — crowned figure with scepter

    # ── Diamonds ♦ (red suit: pips show 1-4 symbols) ──
    "IMG_1659": ("suit", 1, "diamonds"), # 1♦ — single diamond
    "IMG_1646": ("suit", 4, "diamonds"), # 4♦ — four diamonds
    "IMG_1671": ("suit", 5, "diamonds"), # J♦ (Jack) — man with mace, "Wien" / Piatnik shield
    "IMG_1657": ("suit", 6, "diamonds"), # C♦ (Knight) — figure on horseback with sword
    "IMG_1661": ("suit", 8, "diamonds"), # K♦ (King) — crowned bearded figure, fur robe
    # Note: IMG_1666 is a duplicate photo of K♦ — omitted

    # ── Clubs ♣ (black suit: pips show 7-10 symbols) ──
    "IMG_1644": ("suit", 1, "clubs"),    # 7♣ (PIP_1)
    "IMG_1653": ("suit", 2, "clubs"),    # 8♣ (PIP_2)
    "IMG_1663": ("suit", 3, "clubs"),    # 9♣ (PIP_3)
    "IMG_1665": ("suit", 4, "clubs"),    # 10♣ (PIP_4)
    "IMG_1648": ("suit", 6, "clubs"),    # C♣ (Knight) — turban, scimitar, horseback
    # Note: IMG_1647 is a duplicate photo of C♣ — omitted
    # Note: IMG_1664 is a duplicate clubs pip photo — omitted

    # ── Spades ♠ (black suit: pips show 7-10 symbols) ──
    "IMG_1660": ("suit", 1, "spades"),   # 7♠ (PIP_1)
    "IMG_1662": ("suit", 2, "spades"),   # 8♠ (PIP_2)
    "IMG_1678": ("suit", 3, "spades"),   # 9♠ (PIP_3)
    "IMG_1673": ("suit", 6, "spades"),   # C♠ (Knight) — horseback rider with curved sword
    "IMG_1667": ("suit", 7, "spades"),   # Q♠ (Queen) — female figure with flowers
    "IMG_1670": ("suit", 8, "spades"),   # K♠ (King) — crowned figure with scepter
}


def find_card_bbox(img: Image.Image) -> tuple[int, int, int, int]:
    """Find the bounding box of the card (bright region) against a dark background."""
    gray = np.array(img.convert("L"))

    # Adaptive threshold: card pixels are significantly brighter than the wood table
    # Use a percentile-based threshold
    threshold = np.percentile(gray, 60)
    threshold = max(threshold, 100)  # Ensure minimum brightness for card detection

    mask = gray > threshold

    # Clean up noise with erosion/dilation equivalent
    # Find rows/cols that contain card pixels
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        # Fallback: return full image
        return (0, 0, img.width, img.height)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add small margin (2% of dimensions)
    margin_x = int(img.width * 0.01)
    margin_y = int(img.height * 0.01)
    cmin = max(0, cmin - margin_x)
    cmax = min(img.width - 1, cmax + margin_x)
    rmin = max(0, rmin - margin_y)
    rmax = min(img.height - 1, rmax + margin_y)

    return (cmin, rmin, cmax + 1, rmax + 1)


def process_card(heic_path: Path, card_id: str) -> Image.Image:
    """Open HEIC, crop to card, rotate to portrait, resize."""
    img = Image.open(heic_path)
    img = ImageOps.exif_transpose(img)

    # Crop to card boundary
    bbox = find_card_bbox(img)
    cropped = img.crop(bbox)

    # Ensure portrait orientation (height > width)
    if cropped.width > cropped.height:
        cropped = cropped.rotate(90, expand=True)

    # Resize to target dimensions maintaining aspect
    cropped = cropped.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)

    return cropped


def make_filename(card_type: str, value: int, suit: str | None) -> str:
    """Generate a clean filename like tarok_01.jpg or hearts_queen.jpg."""
    if card_type == "tarok":
        return f"tarok_{value:02d}.jpg"

    rank_names = {
        1: "pip1", 2: "pip2", 3: "pip3", 4: "pip4",
        5: "jack", 6: "knight", 7: "queen", 8: "king",
    }
    return f"{suit}_{rank_names[value]}.jpg"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clean old IMG_*.jpg files
    for f in OUT_DIR.glob("IMG_*.jpg"):
        f.unlink()
        print(f"  Removed old {f.name}")

    mapping = {}  # key format: "card_type-value-suit" → "/cards/filename.jpg"
    processed = 0
    skipped = 0

    for heic_file in sorted(SRC_DIR.glob("*.HEIC")):
        stem = heic_file.stem  # e.g., "IMG_1640"

        if stem in GROUP_PHOTOS:
            print(f"  SKIP {stem} (group photo)")
            skipped += 1
            continue

        if stem not in CARD_MAP:
            print(f"  SKIP {stem} (not identified)")
            skipped += 1
            continue

        card_type, value, suit = CARD_MAP[stem]
        out_name = make_filename(card_type, value, suit)
        out_path = OUT_DIR / out_name

        print(f"  {stem}.HEIC → {out_name}  [{card_type} {value} {suit or ''}]")

        img = process_card(heic_file, stem)
        img.save(out_path, "JPEG", quality=92)

        # Build mapping key (matches frontend key format)
        suit_key = suit if suit else "null"
        map_key = f"{card_type}-{value}-{suit_key}"
        mapping[map_key] = f"/cards/{out_name}"
        processed += 1

    # Save mapping JSON
    mapping_path = OUT_DIR / "mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)

    print(f"\nDone: {processed} cards processed, {skipped} skipped")
    print(f"Mapping saved to {mapping_path}")

    # Print TypeScript mapping for cardImages.ts
    print("\n// ── TypeScript mapping for cardImages.ts ──")
    print("const IMAGE_MAP: Record<string, string> = {")
    for key in sorted(mapping.keys()):
        print(f"  '{key}': '{mapping[key]}',")
    print("};")


if __name__ == "__main__":
    main()
