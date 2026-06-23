#!/usr/bin/env python3
"""
Benchmark: NIS-style Lanczos 4x + adaptive sharpen upscaling before OCR.

Compares OCR accuracy WITH and WITHOUT GPU-quality Lanczos upscaling
plus adaptive contrast-aware sharpening (CPU-based, identical algorithm
to the CUDA NIS upscaler used in the plugin).

Usage:
    python benchmark_upscale_ocr.py <card_images_dir>

Input:
    Folder of perspective-corrected VTES card images.
    Filename format: "<card_id>_<card_name>.png" or "<card_name>.png"
    The card name before the extension is used as ground truth.

Output:
    Prints accuracy metrics for both pipelines (with/without upscale).
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import pytesseract
from rapidfuzz.distance import Levenshtein


# ─── Sobel-based name region detector (mirrors C++ detectNameRegion) ───

def detect_name_region(card_bgr: np.ndarray,
                       card_type_hint: str = "") -> np.ndarray:
    h, w = card_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.array([])

    gray = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(2.0, (8, 8))
    enhanced = clahe.apply(gray)

    sobel = cv2.Sobel(enhanced, cv2.CV_16S, 1, 0, 3)
    abs_edges = cv2.convertScaleAbs(sobel)
    _, edge_binary = cv2.threshold(abs_edges, 24, 255, cv2.THRESH_BINARY)

    is_vampire = (card_type_hint == "Vampire")
    if is_vampire:
        zone_top = h * 53 // 100
        zone_bot = h * 60 // 100
    else:
        zone_top = h * 10 // 100
        zone_bot = h * 17 // 100

    zone_top = max(0, zone_top)
    zone_bot = min(h, zone_bot)
    zone_h = zone_bot - zone_top
    if zone_h < 4:
        return np.array([])

    proj = [cv2.countNonZero(edge_binary[zone_top + y, :])
            for y in range(zone_h)]

    smooth = []
    for y in range(zone_h):
        s, c = 0, 0
        for dy in (-1, 0, 1):
            ny = y + dy
            if 0 <= ny < zone_h:
                s += proj[ny]
                c += 1
        smooth.append(s // c if c else 0)

    prefix = [0] * (zone_h + 1)
    for y in range(zone_h):
        prefix[y + 1] = prefix[y] + smooth[y]

    min_name_h = max(8, h * 4 // 100)
    max_name_h = min(zone_h, h * 14 // 100)

    best_start = best_end = best_density = 0
    for start in range(zone_h):
        for cand_h in range(min_name_h, min(max_name_h, zone_h - start) + 1):
            total_edges = prefix[start + cand_h] - prefix[start]
            text_rows = sum(1 for y in range(start, start + cand_h)
                            if smooth[y] > w * 0.02)
            if text_rows < cand_h * 3 // 4:
                continue
            avg_per_row = total_edges // cand_h
            if avg_per_row > best_density:
                best_density = avg_per_row
                best_start = start
                best_end = start + cand_h

    if best_density > 0:
        peak = max(smooth[best_start:best_end])
        trim_thresh = peak * 25 // 100
        while best_start < best_end and smooth[best_start] < trim_thresh:
            best_start += 1
        while best_end > best_start and smooth[best_end - 1] < trim_thresh:
            best_end -= 1
        if best_end - best_start < min_name_h:
            best_density = 0

    if best_density > 0:
        name_top = zone_top + best_start
        name_bottom = zone_top + best_end
    else:
        if is_vampire:
            name_top = h * 53 // 100
            name_bottom = h * 59 // 100
        else:
            name_top = h * 10 // 100
            name_bottom = h * 18 // 100

    name_top = max(0, name_top - 2)
    name_bottom = min(h, name_bottom + 2)

    name_left = int(w * 0.03)
    name_w = int(w * 0.94)
    name_roi = (name_left, name_top, name_w, name_bottom - name_top)

    x, y, rw, rh = name_roi
    x = max(0, x)
    y = max(0, y)
    rw = min(rw, w - x)
    rh = min(rh, h - y)
    if rw < 16 or rh < 4:
        return np.array([])

    return card_bgr[y:y + rh, x:x + rw].copy()


# ─── OCR preprocessing (mirrors C++ preprocessForOcr) ─────────────────

def preprocess_for_ocr(region: np.ndarray,
                       skip_upscale: bool = False) -> np.ndarray:
    if region.size == 0:
        return np.array([])

    if region.ndim == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region.copy()

    clahe = cv2.createCLAHE(2.0, (8, 8))
    enhanced = clahe.apply(gray)

    if skip_upscale or (region.shape[0] > 80 and region.shape[1] > 200):
        upscaled = enhanced
    else:
        upscaled = cv2.resize(enhanced, None, fx=3, fy=3,
                              interpolation=cv2.INTER_LANCZOS4)

    blurred = cv2.GaussianBlur(upscaled, (3, 3), 0)
    binary = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 6)
    denoised = cv2.medianBlur(binary, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
    return denoised


# ─── NIS-style Lanczos 4x + adaptive sharpen (CPU) ────────────────────
# Mirrors the CUDA NIS upscaler kernel: 4x Lanczos-3 resampling followed
# by contrast-adaptive sharpening (stronger in medium-contrast areas).

class NISUpscaler:
    def __init__(self):
        self.scale = 4

    @staticmethod
    def _lanczos(x, a=3):
        if abs(x) >= a:
            return 0.0
        if x == 0.0:
            return 1.0
        pix = np.pi * x
        return np.sin(pix) * np.sin(pix / a) / (pix * pix / a)

    def upscale(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        out_h, out_w = h * self.scale, w * self.scale
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_float = rgb.astype(np.float32) / 255.0
        out = np.zeros((out_h, out_w, 3), dtype=np.float32)

        a = 3
        radius = a
        for y in range(out_h):
            for x in range(out_w):
                sx = (x + 0.5) * w / out_w - 0.5
                sy = (y + 0.5) * h / out_h - 0.5
                ix, iy = int(np.floor(sx)), int(np.floor(sy))
                fx, fy = sx - ix, sy - iy

                for c in range(3):
                    total = 0.0
                    norm = 0.0
                    for dy in range(-radius + 1, radius + 1):
                        sy_ = np.clip(iy + dy, 0, h - 1)
                        wy = self._lanczos(fy - dy, a)
                        for dx in range(-radius + 1, radius + 1):
                            sx_ = np.clip(ix + dx, 0, w - 1)
                            wx = self._lanczos(fx - dx, a)
                            w = wx * wy
                            total += w * rgb_float[sy_, sx_, c]
                            norm += w
                    out[y, x, c] = total / norm if norm > 0 else 0.0

        # Adaptive sharpen (contrast-aware unsharp mask)
        blurred = cv2.blur(out, (5, 5))
        contrast = out - blurred
        abs_contrast = np.abs(contrast)
        gain = np.minimum(abs_contrast * 2.5, 1.0) * 0.45
        sharp = np.clip(out + gain * contrast, 0.0, 1.0)

        out_bgr = cv2.cvtColor((sharp * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return out_bgr


# ─── OCR runner ───────────────────────────────────────────────────────

def run_ocr(image: np.ndarray) -> str:
    text = pytesseract.image_to_string(
        image,
        config='--psm 6 -c tessedit_char_whitelist='
               'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
               '0123456789\',-.!?:/&() ')
    text = text.strip().replace('\n', '').replace('\r', '')
    return text


def sanitize(text: str) -> str:
    """Basic sanity filter: first char uppercase, alpha ratio >= 0.4."""
    text = text.strip()
    if not text:
        return ""
    first = next((c for c in text if not c.isspace()), "")
    if not first or not first.isupper():
        return ""
    total = sum(1 for c in text if not c.isspace())
    alpha = sum(1 for c in text if c.isalpha())
    if total < 3 or total > 55:
        return ""
    if alpha / total < 0.40:
        return ""
    return text


def normalize(s: str) -> str:
    return ''.join(c.lower() for c in s if c.isalnum())


def wer(ref: str, hyp: str) -> float:
    r_words = ref.split()
    h_words = hyp.split()
    d = Levenshtein.distance(r_words, h_words)
    return d / max(len(r_words), 1)


def cer(ref: str, hyp: str) -> float:
    d = Levenshtein.distance(ref, hyp)
    return d / max(len(ref), 1)


@dataclass
class Result:
    filename: str
    ground_truth: str
    baseline_text: str
    upscaled_text: str
    baseline_sanitized: str
    upscaled_sanitized: str
    baseline_time: float
    upscaled_time: float


# ─── Main benchmark ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark NIS-style upscaling for VTES card OCR")
    parser.add_argument("images_dir", help="Directory of card images")
    parser.add_argument("--tesseract-cmd", default="tesseract",
                        help="Tesseract executable path")
    args = parser.parse_args()

    pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    # CPU NIS upscaler (same algorithm as CUDA version: Lanczos 4x + adaptive sharpen)
    upscaler = NISUpscaler()
    print(f"[INFO] NIS upscaler ready (scale={upscaler.scale})")

    # Collect images
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    image_files = sorted(
        [f for f in Path(args.images_dir).iterdir()
         if f.suffix.lower() in exts])

    if not image_files:
        print(f"[ERROR] No images found in {args.images_dir}")
        sys.exit(1)

    print(f"[INFO] Found {len(image_files)} images\n")

    results: List[Result] = []

    for img_path in image_files:
        stem = img_path.stem

        # Ground truth: filename before first underscore is card ID, rest is name
        # or the whole stem is the name
        parts = stem.split("_", 1)
        ground_truth = parts[-1].replace("_", " ").strip()

        card_bgr = cv2.imread(str(img_path))
        if card_bgr is None:
            print(f"[SKIP] Cannot read: {img_path.name}")
            continue

        # Determine type hint from filename (optional)
        type_hint = ""
        if "Vampire" in stem or "vampire" in stem:
            type_hint = "Vampire"

        name_region = detect_name_region(card_bgr, type_hint)
        if name_region.size == 0:
            print(f"[SKIP] No name region: {img_path.name}")
            continue

        # ─── Baseline (no upscale) ─────────────────────────────────
        t0 = time.perf_counter()
        proc_baseline = preprocess_for_ocr(name_region, skip_upscale=False)
        baseline_raw = run_ocr(proc_baseline)
        baseline_time = time.perf_counter() - t0
        baseline_ok = sanitize(baseline_raw)

        # ─── With upscale ──────────────────────────────────────────
        upscaled_text = ""
        upscaled_time = 0.0
        upscaled_ok = ""
        if upscaler:
            t0 = time.perf_counter()
            up_name = upscaler.upscale(name_region)
            up_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            proc_upscaled = preprocess_for_ocr(up_name, skip_upscale=True)
            up_ocr_raw = run_ocr(proc_upscaled)
            upscaled_time = time.perf_counter() - t0 + up_time

            upscaled_ok = sanitize(up_ocr_raw)

        results.append(Result(
            filename=img_path.name,
            ground_truth=ground_truth,
            baseline_text=baseline_raw,
            upscaled_text=upscaled_text,
            baseline_sanitized=baseline_ok,
            upscaled_sanitized=upscaled_ok,
            baseline_time=baseline_time,
            upscaled_time=upscaled_time,
        ))

        sys.stdout.write(".")
        sys.stdout.flush()

    print("\n")

    # ─── Report ────────────────────────────────────────────────────
    n = len(results)

    baseline_exact = sum(1 for r in results
                         if normalize(r.baseline_sanitized) == normalize(r.ground_truth))
    baseline_wer = sum(wer(normalize(r.ground_truth), normalize(r.baseline_sanitized))
                       for r in results if r.baseline_sanitized) / max(n, 1)
    baseline_cer = sum(cer(normalize(r.ground_truth), normalize(r.baseline_sanitized))
                       for r in results if r.baseline_sanitized) / max(n, 1)
    baseline_empty = sum(1 for r in results if not r.baseline_sanitized)
    baseline_avg_time = sum(r.baseline_time for r in results) / max(n, 1)

    print("=" * 60)
    print("BASELINE (OpenCV 3x Lanczos upscale)")
    print("=" * 60)
    print(f"  Exact match:        {baseline_exact}/{n} ({100*baseline_exact/n:.1f}%)")
    print(f"  Avg Word Error Rate: {baseline_wer:.4f}")
    print(f"  Avg Char Error Rate: {baseline_cer:.4f}")
    print(f"  Rejected (sanity):   {baseline_empty}/{n} ({100*baseline_empty/n:.1f}%)")
    print(f"  Avg OCR time:        {baseline_avg_time*1000:.1f} ms")
    print()

    if upscaler and any(r.upscaled_text for r in results):
        up_exact = sum(1 for r in results
                       if normalize(r.upscaled_sanitized) == normalize(r.ground_truth))
        up_wer = sum(wer(normalize(r.ground_truth), normalize(r.upscaled_sanitized))
                     for r in results if r.upscaled_sanitized) / max(n, 1)
        up_cer = sum(cer(normalize(r.ground_truth), normalize(r.upscaled_sanitized))
                     for r in results if r.upscaled_sanitized) / max(n, 1)
        up_empty = sum(1 for r in results if not r.upscaled_sanitized)
        up_avg_time = sum(r.upscaled_time for r in results) / max(n, 1)

        print("=" * 60)
        print("WITH NIS UPSCALE (Lanczos 4x + adaptive sharpen)")
        print("=" * 60)
        print(f"  Exact match:        {up_exact}/{n} ({100*up_exact/n:.1f}%)")
        print(f"  Avg Word Error Rate: {up_wer:.4f}")
        print(f"  Avg Char Error Rate: {up_cer:.4f}")
        print(f"  Rejected (sanity):   {up_empty}/{n} ({100*up_empty/n:.1f}%)")
        print(f"  Avg OCR time:        {up_avg_time*1000:.1f} ms  "
              f"(upscale + OCR)")
        print()

        # Comparison
        print("=" * 60)
        print("DELTA (Upscaled - Baseline)")
        print("=" * 60)
        print(f"  Exact match Δ:      {up_exact - baseline_exact:+d} "
              f"({100*(up_exact-baseline_exact)/n:+.1f}%)")
        print(f"  WER Δ:              {up_wer - baseline_wer:+.4f}")
        print(f"  CER Δ:              {up_cer - baseline_cer:+.4f}")
        print(f"  Rejected Δ:         {up_empty - baseline_empty:+d}")
        print()

        # Detailed per-image comparison
        print("─" * 60)
        print("  DETAILED (sample of mismatches)")
        print("─" * 60)
        mismatches = [r for r in results
                      if normalize(r.baseline_sanitized) != normalize(r.ground_truth)
                      or normalize(r.upscaled_sanitized) != normalize(r.ground_truth)]
        for r in mismatches[:10]:
            gt = r.ground_truth
            b = r.baseline_sanitized or "(rejected)"
            u = r.upscaled_sanitized or "(rejected)"
            print(f"\n  [{r.filename}]")
            print(f"    GT:  {gt}")
            print(f"    REF: {b}")
            print(f"    AI:  {u}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
