"""
dataset/build_mri_dataset.py

Self-contained MRI k-space dataset builder for the TRM-MRI pipeline.
Inspired by the style of the Samsung TRM codebase (argdantic CLI, pydantic
config, numpy .npy outputs, dataset.json metadata) but has zero dependencies
on any other file in that codebase.

Downloads / source:
    fastMRI knee or brain dataset from https://fastmri.org
    Files arrive as .h5 with structure:
        kspace : (num_slices, num_coils, height, width)  complex64
                 or (num_slices, height, width) for single-coil volumes

Output layout:
    <output_dir>/
        train/
            dataset.json         ← metadata (read by your MRI dataloader)
            all__inputs.npy      ← float32  (N, 2, H*W)  [real | imag] undersampled k-space
            all__labels.npy      ← float32  (N, H*W)     RSS image from fully-sampled k-space
            all__masks.npy       ← float32  (N, W)       1 = measured line, 0 = zero-filled
        test/
            dataset.json
            all__inputs.npy
            all__labels.npy
            all__masks.npy
        identifiers.json         ← maps integer index → original filename (for debugging)

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


# ---------------------------------------------------------------------------
# CLI config
# ---------------------------------------------------------------------------

cli = ArgParser()


class DataProcessConfig(BaseModel):
    # Paths
    input_dir:  str = "data/fastmri_knee_raw"
    output_dir: str = "data/mri-knee"

    # Undersampling
    acceleration:    int   = 4     # keep 1/acceleration of phase-encode lines
    center_fraction: float = 0.08  # fraction of central k-space lines always kept

    # Split
    test_fraction: float = 0.1    # fraction of .h5 files held out for test

    # Optional caps (useful for quick smoke-tests)
    max_train_slices: Optional[int] = None
    max_test_slices:  Optional[int] = None

    seed: int = 42


# ---------------------------------------------------------------------------
# Self-contained metadata dataclass
# ---------------------------------------------------------------------------

class MRIDatasetMetadata(BaseModel):
    """
    Written to dataset.json in each split directory.
    Your MRI dataloader reads this to know the shape of the arrays.
    """
    height:          int        # spatial height of each slice (pixels)
    width:           int        # spatial width of each slice (pixels)
    seq_len:         int        # height * width (flattened spatial size)
    num_slices:      int        # total number of examples in this split
    acceleration:    int        # acceleration factor used to build the masks
    center_fraction: float
    is_multicoil:    bool       # True if source data had a coil dimension
    num_files:       int        # number of .h5 source files in this split
    sets:            List[str] = ["all"]


# ---------------------------------------------------------------------------
# k-space helpers
# ---------------------------------------------------------------------------

def rss_reconstruction(kspace: np.ndarray) -> np.ndarray:
    """
    Inverse FFT + root-sum-of-squares coil combination.

    Args:
        kspace : complex64 array, shape (coils, H, W) or (H, W)

    Returns:
        float32 image (H, W), values normalised to [0, 1]
    """
    # Ensure there is always a coil dimension so the rest of the
    # function is identical for single-coil and multi-coil data.
    if kspace.ndim == 2:
        kspace = kspace[np.newaxis]          # (1, H, W)

    # iFFT2 each coil: frequency domain -> image domain
    # ifftshift corrects the quadrant ordering convention so the
    # reconstructed image is not split into four swapped quadrants.
    images = np.fft.ifft2(kspace, axes=(-2, -1))
    images = np.fft.ifftshift(images, axes=(-2, -1))

    # Root-sum-of-squares across coils -> single magnitude image
    rss = np.sqrt((np.abs(images) ** 2).sum(axis=0)).astype(np.float32)

    # Normalise to [0, 1] per slice so pixel values are in a
    # consistent range regardless of the scanner gain setting.
    max_val = rss.max()
    if max_val > 0:
        rss /= max_val

    return rss                               # (H, W) float32


def build_cartesian_mask(width: int,
                         acceleration: int,
                         center_fraction: float,
                         rng: np.random.Generator) -> np.ndarray:
    """
    Random 1-D Cartesian undersampling mask along the phase-encode axis.

    Always keeps the central ACS lines. Randomly samples the remainder
    so that total kept lines ~ width / acceleration.

    Args:
        width           : number of phase-encode lines (k-space columns)
        acceleration    : e.g. 4 means keep ~25% of lines
        center_fraction : fraction of lines always kept at centre
        rng             : seeded numpy Generator for reproducibility

    Returns:
        float32 array (width,)  --  1.0 = measured, 0.0 = zero-filled
    """
    num_keep   = int(round(width / acceleration))
    num_center = int(round(width * center_fraction))
    num_center = min(num_center, num_keep)   # can't exceed total budget
    num_random = max(num_keep - num_center, 0)

    mask = np.zeros(width, dtype=np.float32)

    # Central ACS lines
    center_start = (width - num_center) // 2
    mask[center_start: center_start + num_center] = 1.0

    # Randomly sampled peripheral lines
    peripheral = np.where(mask == 0.0)[0]
    if num_random > 0 and len(peripheral) >= num_random:
        chosen = rng.choice(peripheral, size=num_random, replace=False)
        mask[chosen] = 1.0

    return mask


def apply_mask(kspace: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Zero out k-space lines not present in the mask.

    Args:
        kspace : complex64 (coils, H, W) or (H, W)
        mask   : float32  (W,)

    Returns:
        masked k-space, same shape as input
    """
    # numpy broadcasting applies (W,) across all coils and rows automatically
    return kspace * mask


def kspace_to_input(kspace_masked: np.ndarray) -> np.ndarray:
    """
    Convert masked k-space to a float32 model input array.

    Steps:
        1. Sum across coils (simple coil combination in k-space domain)
        2. Split complex values into real and imaginary channels
        3. Flatten spatial dims: (H, W) -> (H*W,)
        4. Normalise by the maximum complex magnitude
        5. Stack channels: output shape (2, H*W)

    Args:
        kspace_masked : complex64 (coils, H, W) or (H, W)

    Returns:
        float32 (2, H*W)  --  channel 0 = real part, channel 1 = imaginary part
    """
    if kspace_masked.ndim == 2:
        kspace_masked = kspace_masked[np.newaxis]   # (1, H, W)

    # Combine coils by summing in k-space
    combined = kspace_masked.sum(axis=0)            # (H, W) complex

    real_part = combined.real.astype(np.float32).reshape(-1)   # (H*W,)
    imag_part = combined.imag.astype(np.float32).reshape(-1)   # (H*W,)

    # Normalise by max complex magnitude so real and imag stay in proportion
    magnitude = np.sqrt(real_part ** 2 + imag_part ** 2)
    max_mag = magnitude.max()
    if max_mag > 0:
        real_part /= max_mag
        imag_part /= max_mag

    return np.stack([real_part, imag_part], axis=0)             # (2, H*W)


# ---------------------------------------------------------------------------
# .h5 file loading
# ---------------------------------------------------------------------------

def load_h5_slices(filepath: str) -> Tuple[List[np.ndarray], bool]:
    """
    Load all k-space slices from a fastMRI .h5 file.

    Returns:
        slices       : list of per-slice arrays, each (coils, H, W) or (H, W)
        is_multicoil : True if the source file had a coil dimension
    """
    with h5py.File(filepath, "r") as f:
        kspace = f["kspace"][:]     # (slices, [coils,] H, W)

    # Some HDF5 files store complex as a structured dtype with fields "r"/"i"
    if kspace.dtype.names is not None and {"r", "i"} <= set(kspace.dtype.names):
        kspace = kspace["r"] + 1j * kspace["i"]

    kspace = kspace.astype(np.complex64)

    # kspace.ndim == 4  ->  (slices, coils, H, W)  multi-coil
    # kspace.ndim == 3  ->  (slices, H, W)          single-coil
    is_multicoil = (kspace.ndim == 4)

    slices = [kspace[i] for i in range(kspace.shape[0])]
    return slices, is_multicoil


def collect_h5_files(input_dir: str) -> List[str]:
    """Recursively find all .h5 files under input_dir, sorted for determinism."""
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
                   max_slices: Optional[int]) -> Tuple[dict, MRIDatasetMetadata]:
    """
    Process a list of .h5 files into flat numpy arrays ready for training.

    Returns:
        arrays   : dict { "inputs": ndarray, "labels": ndarray, "masks": ndarray }
        metadata : MRIDatasetMetadata  (written to dataset.json)
    """

    # Determine spatial dimensions from the first file
    first_slices, first_is_multicoil = load_h5_slices(h5_files[0])
    first_slice = first_slices[0]

    if first_is_multicoil:
        _, height, width = first_slice.shape    # (coils, H, W)
    else:
        height, width = first_slice.shape       # (H, W)

    print(f"  [{set_name}] spatial size: {height}x{width}, "
          f"{'multi' if first_is_multicoil else 'single'}-coil")

    # Accumulate slices
    all_inputs: List[np.ndarray] = []   # each: (2, H*W) float32
    all_labels: List[np.ndarray] = []   # each: (H*W,)   float32
    all_masks:  List[np.ndarray] = []   # each: (W,)     float32

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

            # Spatial consistency check
            if is_multicoil:
                _, h, w = kspace_slice.shape
            else:
                h, w = kspace_slice.shape

            if h != height or w != width:
                total_skipped += 1
                continue

            # Build a fresh random mask for this slice
            mask = build_cartesian_mask(
                width=width,
                acceleration=config.acceleration,
                center_fraction=config.center_fraction,
                rng=rng,
            )

            # Input: masked k-space -> (2, H*W) float32
            kspace_masked = apply_mask(kspace_slice, mask)
            inp = kspace_to_input(kspace_masked)

            # Label: fully-sampled RSS image -> (H*W,) float32
            label = rss_reconstruction(kspace_slice).reshape(-1)

            all_inputs.append(inp)
            all_labels.append(label)
            all_masks.append(mask)

            total_saved += 1

        if max_slices is not None and total_saved >= max_slices:
            break

    if total_skipped > 0:
        print(f"  [{set_name}] WARNING: skipped {total_skipped} slices "
              f"due to mismatched spatial dimensions.")

    print(f"  [{set_name}] {total_saved} slices saved from {len(h5_files)} files.")

    arrays = {
        "inputs": np.stack(all_inputs, axis=0),   # (N, 2, H*W)
        "labels": np.stack(all_labels, axis=0),   # (N, H*W)
        "masks":  np.stack(all_masks,  axis=0),   # (N, W)
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
# Top-level entry point
# ---------------------------------------------------------------------------

def convert_dataset(config: DataProcessConfig) -> None:

    # Find and shuffle files
    all_files = collect_h5_files(config.input_dir)
    if not all_files:
        raise FileNotFoundError(
            f"No .h5 files found under '{config.input_dir}'.\n"
            "Download the fastMRI dataset from https://fastmri.org and point "
            "--input-dir at the folder containing the .h5 files."
        )

    print(f"Found {len(all_files)} .h5 files in '{config.input_dir}'.")

    rng = np.random.default_rng(config.seed)
    all_files = list(rng.permutation(all_files))    # deterministic shuffle

    # Train / test split
    n_test  = max(1, int(len(all_files) * config.test_fraction))
    n_train = len(all_files) - n_test

    splits = [
        ("train", all_files[:n_train], config.max_train_slices),
        ("test",  all_files[n_train:], config.max_test_slices),
    ]

    print(f"Split: {n_train} train files / {n_test} test files.\n")

    # Process each split
    for set_name, files, max_slices in splits:
        save_dir = os.path.join(config.output_dir, set_name)
        os.makedirs(save_dir, exist_ok=True)

        arrays, metadata = convert_subset(set_name, files, config, rng, max_slices)

        for key, arr in arrays.items():
            path = os.path.join(save_dir, f"all__{key}.npy")
            np.save(path, arr)
            print(f"    {key:10s}  {str(arr.shape):25s}  dtype={arr.dtype}")

        with open(os.path.join(save_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)

        print()

    # Save filename -> index map for debugging
    identifier_list = ["<blank>"] + [os.path.basename(fp) for fp in all_files]
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(identifier_list, f, indent=2)

    print(f"Dataset written to '{config.output_dir}'.")


@cli.command(singleton=True)
def main(config: DataProcessConfig) -> None:
    convert_dataset(config)


if __name__ == "__main__":
    cli()
