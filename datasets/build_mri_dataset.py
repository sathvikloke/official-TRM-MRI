"""
dataset/build_mri_dataset.py

Converts raw fastMRI .h5 files into .npy arrays consumed by the MRI training pipeline.

Output layout (one directory per split, e.g. data/mri-knee/train/):
    all__inputs.npy   float32  (N, 2, H*W)   — real & imag of masked k-space, NOT normalised
    all__labels.npy   float32  (N, H*W)       — RSS magnitude image, normalised to [0, 1]
    all__masks.npy    float32  (N, W)          — 1-D Cartesian undersampling mask
    all__scales.npy   float32  (N,)            — per-slice RSS max used for label normalisation
    dataset.json                               — MRIDatasetMetadata

Why raw (un-normalised) k-space as input?
------------------------------------------
Data consistency enforces:  dc_k = mask * k_obs + (1 - mask) * FFT(pred_image)

For this to work both sides must live in the same physical scale.
The label RSS image is the iFFT of the FULLY-SAMPLED k-space, normalised by its own
max pixel value (stored in all__scales.npy so the model can invert it if needed).
The raw k-space naturally satisfies:  max(|iFFT(k)|) == scale,
so the model can learn to predict images in [0,1] and the FFT of those predictions
will sit in the right numerical range for data-consistency blending.

Usage:
    python -m datasets.build_mri_dataset \\
        --input-dir data/fastmri_knee_raw \\
        --output-dir data/mri-knee \\
        --acceleration 4
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────
# Metadata schema  (self-contained, no dependency on dataset.common)
# ──────────────────────────────────────────────────────────────

class MRIDatasetMetadata(BaseModel):
    """Written to dataset.json; read by MRIDataset in pretrain.py."""
    height: int
    width: int
    seq_len: int          # = height * width
    acceleration: int
    center_fraction: float
    is_multicoil: bool
    total_slices: int
    sets: List[str]


# ──────────────────────────────────────────────────────────────
# CLI config
# ──────────────────────────────────────────────────────────────

class DataProcessConfig(BaseModel):
    input_dir: str = "data/fastmri_knee_raw"
    output_dir: str = "data/mri-knee"
    acceleration: int = 4
    center_fraction: float = 0.08   # fraction of width always kept at centre
    test_fraction: float = 0.1
    seed: int = 42
    max_train_slices: Optional[int] = None
    max_test_slices: Optional[int] = None


cli = ArgParser()


# ──────────────────────────────────────────────────────────────
# MRI physics helpers
# ──────────────────────────────────────────────────────────────

def rss_reconstruction(kspace: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute Root-Sum-of-Squares image from (coils, H, W) complex k-space.
    Returns (image_normalised, scale) where image_normalised ∈ [0, 1].

    Single-coil input (H, W) is handled by inserting a dummy coil dim.

    The scale is the pre-normalisation max pixel value; storing it allows
    the model to reconstruct absolute pixel intensities if needed.
    """
    if kspace.ndim == 2:
        kspace = kspace[np.newaxis]          # (1, H, W)

    # iFFT each coil  →  complex image
    images = np.fft.ifft2(kspace, axes=(-2, -1))
    images = np.fft.ifftshift(images, axes=(-2, -1))

    # RSS combination across coils
    rss = np.sqrt((np.abs(images) ** 2).sum(axis=0)).astype(np.float32)   # (H, W)

    scale = float(rss.max())
    if scale > 0:
        rss /= scale

    return rss, scale


def build_cartesian_mask(width: int, acceleration: int, center_fraction: float,
                          rng: np.random.Generator) -> np.ndarray:
    """
    1-D Cartesian undersampling mask of length `width`.
    Always keeps the central `center_fraction * width` lines.
    Randomly samples the remaining lines to reach `width / acceleration` total.
    """
    num_keep   = max(1, int(round(width / acceleration)))
    num_center = max(1, int(round(width * center_fraction)))
    num_center = min(num_center, num_keep)
    num_random = max(0, num_keep - num_center)

    mask = np.zeros(width, dtype=np.float32)

    # Always-on centre block
    c_start = (width - num_center) // 2
    mask[c_start: c_start + num_center] = 1.0

    # Random peripheral lines
    peripheral = [i for i in range(width) if mask[i] == 0.0]
    if num_random > 0 and len(peripheral) >= num_random:
        chosen = rng.choice(peripheral, size=num_random, replace=False)
        mask[chosen] = 1.0

    return mask


def apply_mask(kspace: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Zero out k-space lines where mask == 0.
    kspace: (coils, H, W) complex  OR  (H, W) complex
    mask:   (W,) float
    Broadcasting zeros entire columns.
    """
    return kspace * mask   # numpy broadcasts (W,) over (..., H, W) correctly


def kspace_to_input(kspace_masked: np.ndarray) -> np.ndarray:
    """
    Convert masked complex k-space → model input array of shape (2, H*W).
    Channel 0: real part (flattened, row-major)
    Channel 1: imaginary part (flattened, row-major)

    NOTE: values are NOT normalised here — raw k-space magnitudes are preserved
    so that data_consistency() inside the model can blend predicted and observed
    k-space on a matching physical scale.
    """
    if kspace_masked.ndim == 3:
        # Multi-coil: sum across coils before splitting real/imag.
        # Simple coil combination in k-space; adequate for a prototype.
        combined = kspace_masked.sum(axis=0)          # (H, W) complex
    else:
        combined = kspace_masked                       # (H, W) complex

    real_part = combined.real.astype(np.float32).reshape(-1)   # (H*W,)
    imag_part = combined.imag.astype(np.float32).reshape(-1)   # (H*W,)

    return np.stack([real_part, imag_part], axis=0)            # (2, H*W)


# ──────────────────────────────────────────────────────────────
# H5 file loading
# ──────────────────────────────────────────────────────────────

def load_h5_slices(filepath: str) -> List[np.ndarray]:
    """
    Load all axial slices from a fastMRI .h5 file.
    Returns a list of complex arrays, each of shape (coils, H, W) or (H, W).
    """
    with h5py.File(filepath, "r") as f:
        kspace = f["kspace"][:]    # (slices, [coils,] H, W)

    # Handle structured dtype (some HDF5 files store complex as {r, i} fields)
    if kspace.dtype.names is not None and {"r", "i"}.issubset(set(kspace.dtype.names)):
        kspace = kspace["r"] + 1j * kspace["i"]

    kspace = kspace.astype(np.complex64)

    # Return one array per slice
    return [kspace[sl] for sl in range(kspace.shape[0])]


# ──────────────────────────────────────────────────────────────
# Per-split conversion
# ──────────────────────────────────────────────────────────────

def convert_subset(
    set_name: str,
    h5_files: List[str],
    config: DataProcessConfig,
    rng: np.random.Generator,
    max_slices: Optional[int],
    height: int,
    width: int,
) -> int:
    """
    Process a list of .h5 files and write the .npy arrays for one split.
    Returns the number of slices written.
    """
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    inputs_list:  List[np.ndarray] = []
    labels_list:  List[np.ndarray] = []
    masks_list:   List[np.ndarray] = []
    scales_list:  List[float]      = []

    skipped = 0
    total   = 0

    for filepath in tqdm(h5_files, desc=f"Processing {set_name}"):
        file_id = Path(filepath).stem
        try:
            slices = load_h5_slices(filepath)
        except Exception as exc:
            print(f"  [WARN] Failed to load {filepath}: {exc}")
            continue

        for sl_kspace in slices:
            if max_slices is not None and total >= max_slices:
                break

            # Determine spatial dims (last two axes)
            h, w = sl_kspace.shape[-2], sl_kspace.shape[-1]

            # Enforce consistent spatial size across all files
            if h != height or w != width:
                skipped += 1
                continue

            # ── Ground-truth label (fully-sampled RSS image) ──────────────
            label_image, scale = rss_reconstruction(sl_kspace)   # (H, W) ∈ [0,1]

            # ── Random Cartesian mask for this slice ──────────────────────
            mask = build_cartesian_mask(width, config.acceleration, config.center_fraction, rng)

            # ── Apply mask to k-space ─────────────────────────────────────
            kspace_masked = apply_mask(sl_kspace, mask)

            # ── Convert to model input ────────────────────────────────────
            inp = kspace_to_input(kspace_masked)   # (2, H*W)

            inputs_list.append(inp)
            labels_list.append(label_image.reshape(-1).astype(np.float32))   # (H*W,)
            masks_list.append(mask)                                            # (W,)
            scales_list.append(scale)

            total += 1

        if max_slices is not None and total >= max_slices:
            break

    if skipped > 0:
        print(f"  [INFO] {set_name}: skipped {skipped} slices with mismatched spatial size "
              f"(expected {height}×{width}).")

    if total == 0:
        raise RuntimeError(f"No slices were written for split '{set_name}'. "
                           f"Check --input-dir and that files have size {height}×{width}.")

    # Stack and save
    np.save(os.path.join(save_dir, "all__inputs.npy"),
            np.stack(inputs_list, axis=0).astype(np.float32))   # (N, 2, H*W)
    np.save(os.path.join(save_dir, "all__labels.npy"),
            np.stack(labels_list, axis=0).astype(np.float32))   # (N, H*W)
    np.save(os.path.join(save_dir, "all__masks.npy"),
            np.stack(masks_list,  axis=0).astype(np.float32))   # (N, W)
    np.save(os.path.join(save_dir, "all__scales.npy"),
            np.array(scales_list, dtype=np.float32))             # (N,)

    # Metadata
    is_multicoil = False
    if h5_files:
        try:
            sample_slices = load_h5_slices(h5_files[0])
            if sample_slices and sample_slices[0].ndim == 3:
                is_multicoil = True
        except Exception:
            pass

    metadata = MRIDatasetMetadata(
        height=height,
        width=width,
        seq_len=height * width,
        acceleration=config.acceleration,
        center_fraction=config.center_fraction,
        is_multicoil=is_multicoil,
        total_slices=total,
        sets=["all"],
    )
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)

    print(f"  [OK] {set_name}: wrote {total} slices → {save_dir}")
    return total


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def _discover_h5_files(input_dir: str) -> List[str]:
    files = sorted(
        str(p) for p in Path(input_dir).rglob("*.h5")
    )
    if not files:
        raise FileNotFoundError(f"No .h5 files found under {input_dir}")
    return files


def _infer_spatial_size(files: List[str]) -> Tuple[int, int]:
    """Read the first file to infer the canonical (H, W)."""
    slices = load_h5_slices(files[0])
    sl = slices[0]
    h, w = sl.shape[-2], sl.shape[-1]
    print(f"  [INFO] Inferred spatial size from first file: {h}×{w}")
    return h, w


def convert_dataset(config: DataProcessConfig) -> None:
    rng = np.random.default_rng(config.seed)

    # Discover all .h5 files
    all_files = _discover_h5_files(config.input_dir)
    print(f"Found {len(all_files)} .h5 files in {config.input_dir}")

    # Infer spatial size from first file
    height, width = _infer_spatial_size(all_files)

    # Train / test split (file-level, not slice-level)
    rng.shuffle(all_files)  # in-place shuffle
    n_test  = max(1, int(round(len(all_files) * config.test_fraction)))
    n_train = len(all_files) - n_test
    train_files = all_files[:n_train]
    test_files  = all_files[n_train:]

    print(f"Split: {n_train} train files, {n_test} test files")

    convert_subset("train", train_files, config, rng,
                   max_slices=config.max_train_slices,
                   height=height, width=width)
    convert_subset("test",  test_files,  config, rng,
                   max_slices=config.max_test_slices,
                   height=height, width=width)


@cli.command(singleton=True)
def main(config: DataProcessConfig) -> None:
    convert_dataset(config)


if __name__ == "__main__":
    cli()
