"""Feature extraction from insect images and segmentation masks."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color, filters, measure, morphology


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def load_bool_mask(path: Path, image_size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as img:
        mask = img.convert("L")
        if image_size is not None:
            mask = mask.resize(image_size, resample=Image.Resampling.NEAREST)
        return np.asarray(mask) > 0


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return int(y_min), int(x_min), int(y_max), int(x_max)


def _safe_ratio(num: float, den: float) -> float:
    if den == 0:
        return np.nan
    return float(num / den)


# ============================================================
# Mandatory features
# ============================================================

def shape_symmetry(mask: np.ndarray) -> float:
    if not mask.any():
        return np.nan
    y_min, x_min, y_max, x_max = _bbox(mask)
    crop = mask[y_min:y_max + 1, x_min:x_max + 1]
    flipped = np.fliplr(crop)
    denom = crop.sum() + flipped.sum()
    if denom == 0:
        return np.nan
    return float(2.0 * np.logical_and(crop, flipped).sum() / denom)


def color_symmetry(rgb: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return np.nan
    y_min, x_min, y_max, x_max = _bbox(mask)
    rgb_crop = rgb[y_min:y_max + 1, x_min:x_max + 1, :3]
    mask_crop = mask[y_min:y_max + 1, x_min:x_max + 1]
    flipped_rgb = np.fliplr(rgb_crop)
    flipped_mask = np.fliplr(mask_crop)
    valid = mask_crop & flipped_mask
    if not valid.any():
        return np.nan
    diffs = np.abs(rgb_crop.astype(float) - flipped_rgb.astype(float))[valid]
    return float(max(0.0, 1.0 - (diffs.mean() / 255.0)))


def bug_pixel_ratio(mask: np.ndarray) -> float:
    return float(np.sum(mask) / mask.size)


def rgb_channel_stats(rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    pixels = rgb[mask]
    features = {}
    for i, prefix in enumerate(["r", "g", "b"]):
        values = pixels[:, i].astype(float)
        features[f"{prefix}_min"] = float(np.min(values))
        features[f"{prefix}_max"] = float(np.max(values))
        features[f"{prefix}_mean"] = float(np.mean(values))
        features[f"{prefix}_median"] = float(np.median(values))
        features[f"{prefix}_std"] = float(np.std(values, ddof=0))
    return features


# ============================================================
# Additional features (improve metrics)
# ============================================================

def shape_features(mask: np.ndarray) -> dict[str, float]:
    features = {}
    labeled = measure.label(mask)
    regions = measure.regionprops(labeled)

    if not regions:
        return {
            "aspect_ratio": np.nan, "circularity": np.nan, "solidity": np.nan,
            "eccentricity": np.nan, "extent": np.nan, "major_axis_length": np.nan,
            "minor_axis_length": np.nan, "equivalent_diameter": np.nan,
        }

    region = max(regions, key=lambda r: r.area)
    r1, c1, r2, c2 = region.bbox
    bbox_w, bbox_h = c2 - c1, r2 - r1
    bbox_area = bbox_w * bbox_h
    perim = measure.perimeter(mask, neighborhood=8)

    features["aspect_ratio"] = _safe_ratio(bbox_w, bbox_h)
    features["circularity"] = _safe_ratio(4 * math.pi * region.area, perim ** 2) if perim > 0 else np.nan
    features["solidity"] = float(region.solidity)
    features["eccentricity"] = float(region.eccentricity)
    features["extent"] = _safe_ratio(region.area, bbox_area)
    features["major_axis_length"] = float(region.axis_major_length)
    features["minor_axis_length"] = float(region.axis_minor_length)
    features["equivalent_diameter"] = float(region.equivalent_diameter_area)

    return features


def hu_moments(mask: np.ndarray) -> dict[str, float]:
    y_min, x_min, y_max, x_max = _bbox(mask)
    crop = mask[y_min:y_max + 1, x_min:x_max + 1].astype(float)
    central = measure.moments_central(crop)
    normalized = measure.moments_normalized(central)
    hu = measure.moments_hu(normalized)
    return {f"hu_moment_{i + 1}": float(v) for i, v in enumerate(hu)}


def hsv_stats(rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    pixels = rgb[mask].astype(np.float32) / 255.0
    hsv = color.rgb2hsv(pixels.reshape(-1, 1, 3)).reshape(-1, 3)
    features = {}
    for i, name in enumerate(["h", "s", "v"]):
        ch = hsv[:, i]
        features[f"hsv_{name}_mean"] = float(np.mean(ch))
        features[f"hsv_{name}_std"] = float(np.std(ch, ddof=0))
    return features


def lab_stats(rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    pixels = rgb[mask].astype(np.float32) / 255.0
    lab = color.rgb2lab(pixels.reshape(-1, 1, 3)).reshape(-1, 3)
    features = {}
    for i, name in enumerate(["l", "a", "b"]):
        ch = lab[:, i]
        features[f"lab_{name}_mean"] = float(np.mean(ch))
        features[f"lab_{name}_std"] = float(np.std(ch, ddof=0))
    return features


def texture_features(rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    y_min, x_min, y_max, x_max = _bbox(mask)
    gray = color.rgb2gray(rgb[y_min:y_max + 1, x_min:x_max + 1].astype(np.float32) / 255.0)
    mask_crop = mask[y_min:y_max + 1, x_min:x_max + 1]
    sobel = filters.sobel(gray)
    inside = sobel[mask_crop]
    if inside.size == 0:
        return {"edge_density": np.nan, "texture_contrast": np.nan}
    threshold = float(np.mean(inside) + np.std(inside))
    return {
        "edge_density": float(np.mean(inside > threshold)),
        "texture_contrast": float(np.std(inside)),
    }


# ============================================================
# Main extraction function
# ============================================================

def extract_features(image_path: Path, mask_path: Path) -> dict[str, float]:
    rgb = load_rgb_image(image_path)
    mask = load_bool_mask(mask_path, image_size=(rgb.shape[1], rgb.shape[0]))

    if not mask.any():
        raise ValueError(f"Empty mask: {mask_path}")

    features = {
        "color_symmetry": color_symmetry(rgb, mask),
        "shape_symmetry": shape_symmetry(mask),
        "bug_pixel_ratio": bug_pixel_ratio(mask),
    }
    features.update(rgb_channel_stats(rgb, mask))
    features.update(shape_features(mask))
    features.update(hu_moments(mask))
    features.update(hsv_stats(rgb, mask))
    features.update(lab_stats(rgb, mask))
    features.update(texture_features(rgb, mask))

    for key, value in list(features.items()):
        if isinstance(value, float) and not np.isfinite(value):
            features[key] = np.nan

    return features
