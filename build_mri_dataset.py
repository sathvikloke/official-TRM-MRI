"""
dataset/build_mri_dataset.py

Self-contained MRI k-space dataset builder for the TRM-MRI pipeline.

Key design decisions:
    1. Label is stored as a COMPLEX image (2 channels: real, imag) normalised
       by scale = max(|kspace|).  This preserves the FFT round-trip exactly:
           FFT(label_real + j*label_imag) == kspace / scale
       so data_consistency can mix the model's predicted k-space with the
       observed k-space without any scale mismatch.

    2. Input k-space is also normalised by the same scale.

    3. A separate magnitude image is NOT stored as the label.  The loss
       function computes MSE on the magnitude of the complex prediction, which
       is what the radiologist actually sees.  PSNR is reported on magnitude.

    4. all__scales.npy stores the per-slice scale factor so predictions can
       be converted back to physical units if needed.

Output layout:
    <output_dir>/
        train/
            dataset.json
            all__inputs.npy    float32  (N, 2, H*W)  normalised [real|imag] masked k-space
            all__labels.npy    float32  (N, 2, H*W)  normalised [real|imag] complex image
            all__masks.npy     float32  (N, W)       1=measured, 0=zero-filled
            all__scales.npy    float32  (N,)         per-slice normalisation factor
        test/  (same layout)
        identifiers.json

Usage:
    python -m dataset.build_mri_dataset \
        --input-dir  data/fastmri_knee_raw \
        --output-dir data/mri-knee \
        --acceleration 4
"""

from typing import Optional, List, Tuple
import os
import json

import h5py
import numpy as np
from tqdm import tqdm

from argdantic import ArgParser
from pydantic import BaseModel


cli = ArgParser()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class DataProcessConfig(BaseModel):
    input_dir:        str   = "data/fastmri_knee_raw"
    output_dir:       str   = "data/mri-knee"
    acceleration:     int   = 4
    center_fraction:  float = 0.08
    test_fraction:    float = 0.1
    max_train_slices: Optional[int] = None
    max_test_slices:  Optional[int] = None
    seed:             int   = 42


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

class MRIDatasetMetadata(BaseModel):
    """Written to dataset.json. Read by MRIDataset in pretrain.py."""
    height:          int
    width:           int
    seq_len:         int        # height * width
    num_slices:      int
    acceleration:    int
    center_fraction: float
    is_multicoil:    bool
    num_files:       int
    sets:            List[str] = ["all"]


# ---------------------------------------------------------------------------
# k-space helpers
# ---------------------------------------------------------------------------

def compute_scale(kspace: np.ndarray) -> float:
    """
    Normalisation scale = max complex magnitude of the fully-sampled k-space.
    Dividing both k-space and the complex image by this ensures:
        FFT(label_complex / scale) == kspace / scale
    which makes data_consistency operate on matching scales.
    """
    if kspace.ndim == 2:
        kspace = kspace[np.newaxis]
    scale = float(np.abs(kspace).max())
    return scale if scale > 0.0 else 1.0


def complex_image_label(kspace: np.ndarray, scale: float) -> np.ndarray:
    """
    Compute the normalised complex image from fully-sampled k-space.

    For multi-coil: sum coil images (complex sum), then normalise.
    This is NOT the same as RSS -- we keep phase so the FFT round-trip works.
    The magnitude of this image is the final reconstruction displayed to users.

    Args:
        kspace : complex64  (coils, H, W)  or  (H, W)
        scale  : float -- max(|kspace|)

    Returns:
        float32  (2, H*W)  channel 0 = real, channel 1 = imag
        FFT of this (as complex) == kspace_coil_sum / scale
    """
    if kspace.ndim == 2:
        kspace = kspace[np.newaxis]                         # (1, H, W)

    # iFFT each coil, sum across coils (complex sum preserves phase)
    images = np.fft.ifft2(kspace, axes=(-2, -1))           # (coils, H, W) complex
    images = np.fft.ifftshift(images, axes=(-2, -1))
    combined = images.sum(axis=0)                           # (H, W) complex

    # Normalise by scale
    combined = combined / scale                             # (H, W) complex

    real_part = combined.real.astype(np.float32).reshape(-1)   # (H*W,)
    imag_part = combined.imag.astype(np.float32).reshape(-1)   # (H*W,)

    return np.stack([real_part, imag_part], axis=0)             # (2, H*W)


def kspace_to_input(kspace_masked: np.ndarray, scale: float) -> np.ndarray:
    """
    Convert masked k-space to model input, normalised by scale.

    Args:
        kspace_masked : complex64  (coils, H, W)  or  (H, W)
        scale         : float -- same scale used in complex_image_label

    Returns:
        float32  (2, H*W)   channel 0 = real, channel 1 = imaginary
    """
    if kspace_masked.ndim == 2:
        kspace_masked = kspace_masked[np.newaxis]

    # Sum coils in k-space domain
    combined = kspace_masked.sum(axis=0)                    # (H, W) complex

    real_part = (combined.real / scale).astype(np.float32).reshape(-1)
    imag_part = (combined.imag / scale).astype(np.float32).reshape(-1)

    return np.stack([real_part, imag_part], axis=0)         # (2, H*W)


def build_cartesian_mask(width: int,
                         acceleration: int,
                         center_fraction: float,
                         rng: np.random.Generator) -> np.ndarray:
    """
    Random 1-D Cartesian undersampling mask.
    Returns float32 (width,) with 1.0=measured, 0.0=zero-filled.
    """
    num_keep   = int(round(width / acceleration))
    num_center = int(round(width * center_fraction))
    num_center = min(num_center, num_keep)
    num_random = max(num_keep - num_center, 0)

    mask = np.zeros(width, dtype=np.float32)
    cs   = (width - num_center) // 2
    mask[cs: cs + num_center] = 1.0

    peripheral = np.where(mask == 0.0)[0]
    if num_random > 0 and len(peripheral) >= num_random:
        chosen = rng.choice(peripheral, size=num_random, replace=False)
        mask[chosen] = 1.0

    return mask


def apply_mask(kspace: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out k-space lines not in the mask."""
    return kspace * mask


# ---------------------------------------------------------------------------
# .h5 loading
# ---------------------------------------------------------------------------

def load_h5_slices(filepath: str) -> Tuple[List[np.ndarray], bool]:
    with h5py.File(filepath, "r") as f:
        kspace = f["kspace"][:]

    if kspace.dtype.names is not None and {"r", "i"} <= set(kspace.dtype.names):
        kspace = kspace["r"] + 1j * kspace["i"]

    kspace = kspace.astype(np.complex64)
    is_multicoil = (kspace.ndim == 4)
    slices = [kspace[i] for i in range(kspace.shape[0])]
    return slices, is_multicoil


def collect_h5_files(input_dir: str) -> List[str]:
    files = []
    for root, _, fnames in os.walk(input_dir):
        for fname in sorted(fnames):
            if fname.endswith(".h5"):
                files.append(os.path.join(root, fname))
    return files


# ---------------------------------------------------------------------------
# Per-split conversion
# ---------------------------------------------------------------------------

def convert_subset(set_name:   str,
                   h5_files:   List[str],
                   config:     DataProcessConfig,
                   rng:        np.random.Generator,
                   max_slices: Optional[int],
                   ) -> Tuple[dict, MRIDatasetMetadata]:

    first_slices, first_is_multicoil = load_h5_slices(h5_files[0])
    first_slice = first_slices[0]

    if first_is_multicoil:
        _, height, width = first_slice.shape
    else:
        height, width = first_slice.shape

    print(f"  [{set_name}] {height}x{width}  "
          f"({'multi' if first_is_multicoil else 'single'}-coil)")

    all_inputs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_masks:  List[np.ndarray] = []
    all_scales: List[float]      = []

    total_saved   = 0
    total_skipped = 0

    for filepath in tqdm(h5_files, desc=f"[{set_name}]"):
        try:
            slices, is_multicoil = load_h5_slices(filepath)
        except Exception as exc:
            print(f"  WARNING: skipping {filepath} -- {exc}")
            continue

        for kspace_slice in slices:
            if max_slices is not None and total_saved >= max_slices:
                break

            if is_multicoil:
                _, h, w = kspace_slice.shape
            else:
                h, w = kspace_slice.shape

            if h != height or w != width:
                total_skipped += 1
                continue

            # Per-slice normalisation scale
            scale = compute_scale(kspace_slice)

            # Mask
            mask = build_cartesian_mask(width, config.acceleration,
                                        config.center_fraction, rng)

            # Input: masked k-space / scale  -> (2, H*W)
            kspace_masked = apply_mask(kspace_slice, mask)
            inp = kspace_to_input(kspace_masked, scale)

            # Label: complex image / scale   -> (2, H*W)
            # FFT(label_real + j*label_imag) == kspace_coilsum / scale exactly
            label = complex_image_label(kspace_slice, scale)

            all_inputs.append(inp)
            all_labels.append(label)
            all_masks.append(mask)
            all_scales.append(scale)
            total_saved += 1

        if max_slices is not None and total_saved >= max_slices:
            break

    if total_skipped > 0:
        print(f"  [{set_name}] WARNING: skipped {total_skipped} slices "
              f"(spatial dim mismatch).")

    print(f"  [{set_name}] {total_saved} slices saved from {len(h5_files)} files.")

    arrays = {
        "inputs": np.stack(all_inputs, axis=0),           # (N, 2, H*W)
        "labels": np.stack(all_labels, axis=0),           # (N, 2, H*W)  complex image
        "masks":  np.stack(all_masks,  axis=0),           # (N, W)
        "scales": np.array(all_scales, dtype=np.float32), # (N,)
    }

    metadata = MRIDatasetMetadata(
        height=height,
        width=width,
        seq_len=height * width,
        num_slices=total_saved,
        acceleration=config.acceleration,
        center_fraction=config.center_fraction,
        is_multicoil=first_is_multicoil,
        num_files=len(h5_files),
    )

    return arrays, metadata


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def convert_dataset(config: DataProcessConfig) -> None:
    all_files = collect_h5_files(config.input_dir)
    if not all_files:
        raise FileNotFoundError(
            f"No .h5 files found under '{config.input_dir}'.\n"
            "Download from https://fastmri.org."
        )

    print(f"Found {len(all_files)} .h5 files.")

    rng = np.random.default_rng(config.seed)
    all_files = list(rng.permutation(all_files))

    n_test  = max(1, int(len(all_files) * config.test_fraction))
    n_train = len(all_files) - n_test
    splits  = [
        ("train", all_files[:n_train], config.max_train_slices),
        ("test",  all_files[n_train:], config.max_test_slices),
    ]

    print(f"Split: {n_train} train / {n_test} test files.\n")

    for set_name, files, max_slices in splits:
        save_dir = os.path.join(config.output_dir, set_name)
        os.makedirs(save_dir, exist_ok=True)

        arrays, metadata = convert_subset(set_name, files, config, rng, max_slices)

        for key, arr in arrays.items():
            np.save(os.path.join(save_dir, f"all__{key}.npy"), arr)
            print(f"    {key:10s}  {str(arr.shape):25s}  dtype={arr.dtype}")

        with open(os.path.join(save_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)
        print()

    identifier_list = ["<blank>"] + [os.path.basename(fp) for fp in all_files]
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(identifier_list, f, indent=2)

    print(f"Done. Dataset written to '{config.output_dir}'.")


@cli.command(singleton=True)
def main(config: DataProcessConfig) -> None:
    convert_dataset(config)


if __name__ == "__main__":
    cli()
