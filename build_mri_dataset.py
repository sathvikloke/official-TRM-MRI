"""
dataset/build_mri_dataset.py

MRI k-space dataset builder for the TRM pipeline.
Modeled after dataset/build_sudoku_dataset.py.

Loads fastMRI .h5 files, generates random Cartesian undersampling masks,
and saves inputs/labels/metadata in the same .npy format the rest of the
TRM pipeline already expects.

Usage:
    python -m dataset.build_mri_dataset \
        --input-dir data/fastmri_knee_raw \
        --output-dir data/mri-knee \
        --acceleration 4

Output layout (mirrors sudoku/maze builders):
    data/mri-knee/
        train/
            dataset.json
            all__inputs.npy          # (N, 2, H*W)  float32  [real, imag] of undersampled k-space
            all__labels.npy          # (N, H*W)      float32  RSS reconstruction of fully-sampled k-space
            all__masks.npy           # (N, W)        float32  1 = keep, 0 = zero-fill  (line mask)
            all__puzzle_identifiers.npy
            all__puzzle_indices.npy
            all__group_indices.npy
        test/
            dataset.json
            all__inputs.npy
            all__labels.npy
            all__masks.npy
            all__puzzle_identifiers.npy
            all__puzzle_indices.npy
            all__group_indices.npy
        identifiers.json
"""

from typing import Optional, List, Tuple
import os
import json
import math

import h5py
import numpy as np
from tqdm import tqdm

from argdantic import ArgParser
from pydantic import BaseModel

# Reuse the shared metadata dataclass so pretrain.py / puzzle_dataset.py are happy
from dataset.common import PuzzleDatasetMetadata


# ---------------------------------------------------------------------------
# CLI config
# ---------------------------------------------------------------------------

cli = ArgParser()


class DataProcessConfig(BaseModel):
    input_dir: str = "data/fastmri_knee_raw"
    output_dir: str = "data/mri-knee"

    # Cartesian acceleration factor: keep 1/acceleration of phase-encode lines
    acceleration: int = 4

    # Fraction of files reserved for the test split
    test_fraction: float = 0.1

    # Centre fraction of k-space always kept (ACS / autocalibration signal)
    center_fraction: float = 0.08

    # Cap total slices per split (None = use everything)
    max_train_slices: Optional[int] = None
    max_test_slices: Optional[int] = None

    seed: int = 42


# ---------------------------------------------------------------------------
# k-space helpers
# ---------------------------------------------------------------------------

def rss_reconstruction(kspace: np.ndarray) -> np.ndarray:
    """
    Root-sum-of-squares reconstruction from multi-coil or single-coil k-space.

    Args:
        kspace: complex64 array of shape (coils, H, W) or (H, W).

    Returns:
        float32 image of shape (H, W), normalised to [0, 1].
    """
    if kspace.ndim == 2:
        # Single-coil — add a fake coil dimension so the rest of the code is uniform
        kspace = kspace[np.newaxis]

    # iFFT2 each coil independently
    images = np.fft.ifft2(kspace, axes=(-2, -1))          # (coils, H, W) complex
    images = np.fft.ifftshift(images, axes=(-2, -1))

    # Root-sum-of-squares across coils
    rss = np.sqrt((np.abs(images) ** 2).sum(axis=0))       # (H, W) float32

    # Normalise to [0, 1] per slice
    rss = rss.astype(np.float32)
    max_val = rss.max()
    if max_val > 0:
        rss /= max_val

    return rss


def build_cartesian_mask(width: int,
                         acceleration: int,
                         center_fraction: float,
                         rng: np.random.Generator) -> np.ndarray:
    """
    Random 1-D Cartesian undersampling mask (along the phase-encode / W axis).

    Always retains the central ACS lines. Randomly samples the remainder so
    that total kept lines ≈ width / acceleration.

    Returns:
        float32 array of shape (width,), values in {0.0, 1.0}.
    """
    num_keep = int(round(width / acceleration))
    num_center = int(round(width * center_fraction))
    num_center = min(num_center, num_keep)          # can't keep more than budget
    num_random = max(num_keep - num_center, 0)

    mask = np.zeros(width, dtype=np.float32)

    # Central lines
    center_start = (width - num_center) // 2
    mask[center_start: center_start + num_center] = 1.0

    # Random peripheral lines
    peripheral_indices = [i for i in range(width) if mask[i] == 0.0]
    if num_random > 0 and len(peripheral_indices) >= num_random:
        chosen = rng.choice(peripheral_indices, size=num_random, replace=False)
        mask[chosen] = 1.0

    return mask


def apply_mask_to_kspace(kspace: np.ndarray,
                         mask: np.ndarray) -> np.ndarray:
    """
    Apply a 1-D line mask to k-space.

    Args:
        kspace: complex64 array (coils, H, W) or (H, W).
        mask:   float32 array (W,).

    Returns:
        masked k-space, same shape as input.
    """
    # Broadcast mask across coils and height
    return kspace * mask   # numpy broadcasting handles (coils,H,W) * (W,)


def kspace_to_input_array(kspace_masked: np.ndarray,
                           height: int,
                           width: int) -> np.ndarray:
    """
    Convert masked k-space (possibly multi-coil) to a model-ready float32 array.

    For multi-coil: sum real and imaginary parts across coils then stack.
    This collapses coil dimension while preserving real/imag structure.

    Returns:
        float32 (2, H*W)  — channel 0 = real, channel 1 = imaginary.
    """
    if kspace_masked.ndim == 2:
        kspace_masked = kspace_masked[np.newaxis]   # (1, H, W)

    # Sum coils  (simple coil-combination in k-space; richer options can be added later)
    combined = kspace_masked.sum(axis=0)            # (H, W) complex

    real_part = combined.real.astype(np.float32).reshape(-1)   # (H*W,)
    imag_part = combined.imag.astype(np.float32).reshape(-1)   # (H*W,)

    # Normalise by the max magnitude so values are in a consistent range
    max_mag = np.sqrt(real_part ** 2 + imag_part ** 2).max()
    if max_mag > 0:
        real_part /= max_mag
        imag_part /= max_mag

    return np.stack([real_part, imag_part], axis=0)             # (2, H*W)


# ---------------------------------------------------------------------------
# .h5 loading
# ---------------------------------------------------------------------------

def load_h5_slices(filepath: str) -> List[np.ndarray]:
    """
    Load all k-space slices from a fastMRI .h5 file.

    fastMRI .h5 structure:
        kspace  : (num_slices, num_coils, height, width)  complex64
                  OR (num_slices, height, width) for single-coil files.

    Returns a list of per-slice complex arrays of shape (coils, H, W) or (H, W).
    """
    slices = []
    with h5py.File(filepath, "r") as f:
        kspace = f["kspace"][:]         # load everything into RAM

    # kspace dtype may be stored as real+imag pair; h5py surfaces this as
    # structured array with fields 'r' and 'i', or as complex directly.
    if kspace.dtype.names is not None and set(kspace.dtype.names) >= {"r", "i"}:
        kspace = kspace["r"] + 1j * kspace["i"]
    kspace = kspace.astype(np.complex64)

    for sl_idx in range(kspace.shape[0]):
        slices.append(kspace[sl_idx])   # (coils, H, W) or (H, W)

    return slices


def collect_h5_files(input_dir: str) -> List[str]:
    """Recursively collect all .h5 files under input_dir."""
    files = []
    for root, _, fnames in os.walk(input_dir):
        for fname in sorted(fnames):
            if fname.endswith(".h5"):
                files.append(os.path.join(root, fname))
    return files


# ---------------------------------------------------------------------------
# Per-split conversion
# ---------------------------------------------------------------------------

def convert_subset(set_name: str,
                   h5_files: List[str],
                   config: DataProcessConfig,
                   rng: np.random.Generator,
                   max_slices: Optional[int]) -> Tuple[dict, PuzzleDatasetMetadata]:
    """
    Process a list of .h5 files into the TRM .npy dataset format.

    Returns (results_dict, metadata).
    """
    # ---- figure out spatial dimensions from the first file ----------------
    with h5py.File(h5_files[0], "r") as f:
        sample_kspace = f["kspace"][0]          # first slice
    if sample_kspace.dtype.names is not None and set(sample_kspace.dtype.names) >= {"r", "i"}:
        sample_kspace = sample_kspace["r"] + 1j * sample_kspace["i"]

    if sample_kspace.ndim == 3:
        _, height, width = sample_kspace.shape  # multi-coil
    else:
        height, width = sample_kspace.shape     # single-coil

    seq_len_1d = height * width

    # ---- TRM pipeline arrays ----------------------------------------------
    results = {k: [] for k in [
        "inputs",           # (N, 2, H*W)  float32
        "labels",           # (N, H*W)     float32
        "masks",            # (N, W)       float32
        "puzzle_identifiers",
        "puzzle_indices",
        "group_indices",
    ]}

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    example_id = 0
    puzzle_id = 0
    total_slices_saved = 0

    for file_id, filepath in enumerate(tqdm(h5_files, desc=f"[{set_name}] loading .h5 files")):
        try:
            slices = load_h5_slices(filepath)
        except Exception as exc:
            print(f"  WARNING: could not load {filepath}: {exc}  — skipping.")
            continue

        for kspace_slice in slices:
            if max_slices is not None and total_slices_saved >= max_slices:
                break

            # Spatial consistency check
            if kspace_slice.ndim == 3:
                _, h, w = kspace_slice.shape
            else:
                h, w = kspace_slice.shape

            if h != height or w != width:
                # Skip slices whose spatial dims differ from the first file
                continue

            # Build undersampling mask (new random mask per slice)
            mask_1d = build_cartesian_mask(
                width=width,
                acceleration=config.acceleration,
                center_fraction=config.center_fraction,
                rng=rng,
            )                                               # (W,)

            # Apply mask
            kspace_masked = apply_mask_to_kspace(kspace_slice, mask_1d)

            # Input: (2, H*W) float32
            inp = kspace_to_input_array(kspace_masked, height, width)

            # Label: RSS image (H*W,) float32 from FULLY-sampled k-space
            label_image = rss_reconstruction(kspace_slice)  # (H, W)
            label_flat = label_image.reshape(-1)            # (H*W,)

            results["inputs"].append(inp)
            results["labels"].append(label_flat)
            results["masks"].append(mask_1d)

            # Each slice is its own "puzzle" (one example per group, like Sudoku)
            results["puzzle_identifiers"].append(file_id)
            example_id += 1
            puzzle_id += 1
            results["puzzle_indices"].append(example_id)
            results["group_indices"].append(puzzle_id)

            total_slices_saved += 1

        if max_slices is not None and total_slices_saved >= max_slices:
            break

    # ---- Stack and save ---------------------------------------------------
    print(f"  [{set_name}] saved {total_slices_saved} slices from {len(h5_files)} files.")

    stacked = {
        "inputs":  np.stack(results["inputs"],  axis=0).astype(np.float32),  # (N, 2, H*W)
        "labels":  np.stack(results["labels"],  axis=0).astype(np.float32),  # (N, H*W)
        "masks":   np.stack(results["masks"],   axis=0).astype(np.float32),  # (N, W)

        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
        "puzzle_indices":     np.array(results["puzzle_indices"],     dtype=np.int32),
        "group_indices":      np.array(results["group_indices"],      dtype=np.int32),
    }

    # ---- Metadata (matches PuzzleDatasetMetadata schema) ------------------
    #
    # vocab_size / pad_id / ignore_label_id are not meaningful for continuous
    # MRI data; we set them to sentinel values so the shared dataclass
    # validates without complaints.
    #
    # seq_len encodes the 1-D size of one sample EXCLUDING the channel dim.
    # The MRI model reads inputs.shape directly, so this is informational.
    metadata = PuzzleDatasetMetadata(
        seq_len=seq_len_1d,                     # H * W
        vocab_size=1,                           # not used (continuous data)
        pad_id=0,
        ignore_label_id=None,
        blank_identifier_id=0,
        num_puzzle_identifiers=len(h5_files) + 1,   # +1 for blank
        total_groups=total_slices_saved,
        mean_puzzle_examples=1.0,
        total_puzzles=total_slices_saved,
        sets=["all"],
    )

    return stacked, metadata


# ---------------------------------------------------------------------------
# Top-level conversion
# ---------------------------------------------------------------------------

def convert_dataset(config: DataProcessConfig) -> None:
    rng = np.random.default_rng(config.seed)

    # Collect all .h5 files and shuffle deterministically
    all_files = collect_h5_files(config.input_dir)
    if not all_files:
        raise FileNotFoundError(
            f"No .h5 files found under '{config.input_dir}'. "
            "Please point --input-dir at the fastMRI download directory."
        )

    print(f"Found {len(all_files)} .h5 files in '{config.input_dir}'.")

    rng.shuffle(all_files)                      # in-place shuffle

    # Train / test split
    n_test  = max(1, int(len(all_files) * config.test_fraction))
    n_train = len(all_files) - n_test

    train_files = all_files[:n_train]
    test_files  = all_files[n_train:]

    print(f"Split: {len(train_files)} train files, {len(test_files)} test files.")

    splits = [
        ("train", train_files, config.max_train_slices),
        ("test",  test_files,  config.max_test_slices),
    ]

    identifier_list = ["<blank>"]   # index 0 = blank

    for set_name, files, max_slices in splits:
        save_dir = os.path.join(config.output_dir, set_name)
        os.makedirs(save_dir, exist_ok=True)

        stacked, metadata = convert_subset(set_name, files, config, rng, max_slices)

        # Save numpy arrays
        for key, arr in stacked.items():
            np.save(os.path.join(save_dir, f"all__{key}.npy"), arr)
            print(f"  Saved {key}: {arr.shape}  dtype={arr.dtype}")

        # Save dataset metadata JSON (read by PuzzleDataset)
        with open(os.path.join(save_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)

        # Track file identifiers (for debugging / visualisation)
        for fp in files:
            identifier_list.append(os.path.basename(fp))

    # Save global identifier map
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(identifier_list, f, indent=2)

    print(f"\nDone. Dataset written to '{config.output_dir}'.")

    # Print a summary of what the model will see
    sample_inp_path = os.path.join(config.output_dir, "train", "all__inputs.npy")
    if os.path.exists(sample_inp_path):
        sample = np.load(sample_inp_path, mmap_mode="r")
        print(f"  inputs  shape : {sample.shape}   (N, 2, H*W)")
        print(f"  Example acceleration ~{config.acceleration}x => ~{100 // config.acceleration}% of k-space lines kept.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@cli.command(singleton=True)
def main(config: DataProcessConfig) -> None:
    convert_dataset(config)


if __name__ == "__main__":
    cli()
