"""
pretrain.py  —  TRM-MRI training script.

Changes from the original Samsung TRM pretrain.py:
  1. PuzzleDataset replaced by MRIDataset (reads all__inputs/labels/masks.npy).
  2. create_model() reads height and width from dataset.json and passes them to
     model config so TinyRecursiveReasoningModel_MRIConfig is satisfied.
  3. Optimizer branch: puzzle_emb_ndim is always 0 for MRI, so only AdamATan2 is used.
  4. Everything else — distributed training, EMA, W&B, cosine LR, checkpointing — unchanged.
"""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import coolname
import hydra
import numpy as np
import pydantic
import torch
import torch.distributed as dist
import tqdm
import wandb
from adam_atan2 import AdamATan2
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from models.ema import EMAHelper
from utils.functions import get_model_source_path, load_model_class


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str
    loss: LossConfig


class PretrainConfig(pydantic.BaseModel):
    arch: ArchConfig
    data_paths: List[str]
    data_paths_test: List[str] = []

    global_batch_size: int = 4
    epochs: int = 50000

    lr: float = 1e-4
    lr_min_ratio: float = 1.0
    lr_warmup_steps: int = 2000

    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95

    # Unused for MRI but kept for CLI compat
    puzzle_emb_lr: float = 1e-2
    puzzle_emb_weight_decay: float = 0.1

    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: int = 0
    eval_save_outputs: List[str] = []

    ema: bool = False
    ema_rate: float = 0.999
    freeze_weights: bool = False


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any
    step: int
    total_steps: int


# ──────────────────────────────────────────────────────────────────────────────
# MRI Dataset  (self-contained, no puzzle_dataset dependency)
# ──────────────────────────────────────────────────────────────────────────────

class MRIDatasetMetadata:
    """Thin wrapper around the dataset.json fields we actually need."""
    def __init__(self, d: dict) -> None:
        self.height:       int   = d["height"]
        self.width:        int   = d["width"]
        self.seq_len:      int   = d["seq_len"]
        self.total_slices: int   = d["total_slices"]
        self.sets:         List[str] = d["sets"]

        # Fields expected by the original pretrain.py scaffolding
        self.total_groups:        int   = d["total_slices"]
        self.mean_puzzle_examples: float = 1.0


class MRIDatasetConfig(pydantic.BaseModel):
    seed:             int
    dataset_paths:    List[str]
    global_batch_size: int
    test_set_mode:    bool
    epochs_per_iter:  int
    rank:             int
    num_replicas:     int


class MRIDataset(IterableDataset):
    """
    Reads all__inputs.npy, all__labels.npy, all__masks.npy and yields
    (set_name, batch_dict, effective_global_batch_size) tuples — the same
    interface as the original PuzzleDataset.

    Array shapes on disk:
        inputs  (N, 2, H*W)  float32
        labels  (N, H*W)     float32
        masks   (N, W)       float32
    """

    def __init__(self, config: MRIDatasetConfig, split: str = "train") -> None:
        super().__init__()
        self.config = config
        self.split  = split

        assert config.global_batch_size % config.num_replicas == 0
        self.local_batch_size = config.global_batch_size // config.num_replicas

        # Load and merge metadata across dataset paths
        all_meta: List[MRIDatasetMetadata] = []
        for p in config.dataset_paths:
            json_path = os.path.join(p, split, "dataset.json")
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"dataset.json not found: {json_path}")
            with open(json_path) as f:
                all_meta.append(MRIDatasetMetadata(json.load(f)))

        # Validate consistency
        for m in all_meta[1:]:
            assert m.height == all_meta[0].height and m.width == all_meta[0].width, \
                "All dataset paths must have the same spatial dimensions."

        self.metadata = all_meta[0]
        self.metadata.total_slices        = sum(m.total_slices for m in all_meta)
        self.metadata.total_groups        = self.metadata.total_slices
        self.metadata.mean_puzzle_examples = 1.0

        # Lazy-loaded arrays
        self._data: Optional[Dict[str, Dict[str, np.ndarray]]] = None
        self._iters = 0

    def _lazy_load(self) -> None:
        if self._data is not None:
            return
        self._data = {}
        for i, path in enumerate(self.config.dataset_paths):
            key = "all" if i == 0 else f"all{i}"
            split_dir = os.path.join(path, self.split)
            self._data[key] = {
                "inputs": np.load(os.path.join(split_dir, "all__inputs.npy"), mmap_mode="r"),
                "labels": np.load(os.path.join(split_dir, "all__labels.npy"), mmap_mode="r"),
                "masks":  np.load(os.path.join(split_dir, "all__masks.npy"),  mmap_mode="r"),
            }

    def _collate(self, inputs: np.ndarray, labels: np.ndarray, masks: np.ndarray) -> Dict[str, torch.Tensor]:
        return {
            "inputs": torch.from_numpy(inputs.copy()).float(),
            "labels": torch.from_numpy(labels.copy()).float(),
            "masks":  torch.from_numpy(masks.copy()).float(),
        }

    # ── Test iterator: sequential, complete coverage ─────────────────────

    def _iter_test(self) -> Iterator[Tuple[str, Dict[str, torch.Tensor], int]]:
        for set_name, dataset in self._data.items():   # type: ignore
            N = len(dataset["inputs"])
            start = 0
            while start < N:
                end = min(N, start + self.config.global_batch_size)
                # Drop partial last batch to avoid distributed allreduce hang
                if end - start < self.config.global_batch_size:
                    break

                lo = start + self.config.rank * self.local_batch_size
                hi = start + (self.config.rank + 1) * self.local_batch_size

                batch = self._collate(
                    dataset["inputs"][lo:hi],
                    dataset["labels"][lo:hi],
                    dataset["masks"][lo:hi],
                )
                yield set_name, batch, end - start
                start += self.config.global_batch_size

    # ── Train iterator: randomly shuffled, epochs_per_iter epochs ────────

    def _iter_train(self) -> Iterator[Tuple[str, Dict[str, torch.Tensor], int]]:
        for set_name, dataset in self._data.items():   # type: ignore
            self._iters += 1
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))

            N = len(dataset["inputs"])
            # Concatenate permutations from multiple epochs
            all_indices = np.concatenate(
                [rng.permutation(N) for _ in range(self.config.epochs_per_iter)]
            )

            start = 0
            while start + self.config.global_batch_size <= len(all_indices):
                global_idx = all_indices[start: start + self.config.global_batch_size]

                lo = self.config.rank * self.local_batch_size
                hi = (self.config.rank + 1) * self.local_batch_size
                local_idx = global_idx[lo:hi]

                batch = self._collate(
                    dataset["inputs"][local_idx],
                    dataset["labels"][local_idx],
                    dataset["masks"][local_idx],
                )
                yield set_name, batch, self.config.global_batch_size
                start += self.config.global_batch_size

    def __iter__(self) -> Iterator[Tuple[str, Dict[str, torch.Tensor], int]]:
        self._lazy_load()
        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()


# ──────────────────────────────────────────────────────────────────────────────
# Dataloader factory
# ──────────────────────────────────────────────────────────────────────────────

def create_dataloader(
    config: PretrainConfig,
    split: str,
    rank: int,
    world_size: int,
    **kwargs,
) -> Tuple[DataLoader, MRIDatasetMetadata]:
    paths = config.data_paths_test if config.data_paths_test and split == "test" else config.data_paths
    dataset = MRIDataset(
        MRIDatasetConfig(
            seed=config.seed,
            dataset_paths=paths,
            rank=rank,
            num_replicas=world_size,
            **kwargs,
        ),
        split=split,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True,
    )
    return loader, dataset.metadata


# ──────────────────────────────────────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────────────────────────────────────

def create_model(
    config: PretrainConfig,
    train_metadata: MRIDatasetMetadata,
    rank: int,
    world_size: int,
) -> Tuple[nn.Module, List[torch.optim.Optimizer], List[float]]:

    model_cfg = dict(
        **config.arch.__pydantic_extra__,       # type: ignore  (YAML extra fields)
        batch_size=config.global_batch_size // world_size,
        # Spatial dimensions — the MRI model needs these
        height=train_metadata.height,
        width=train_metadata.width,
        # Compat keys passed by pretrain convention; MRIConfig accepts and ignores them
        vocab_size=1,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=1,
        causal=False,
    )

    model_cls    = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)   # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)   # type: ignore

        if rank == 0:
            _load_checkpoint(model, config)

        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # MRI always uses puzzle_emb_ndim=0, so only one optimizer
    optimizers   = [AdamATan2(model.parameters(), lr=0, weight_decay=config.weight_decay,
                               betas=(config.beta1, config.beta2))]
    optimizer_lrs = [config.lr]

    return model, optimizers, optimizer_lrs


# ──────────────────────────────────────────────────────────────────────────────
# LR schedule
# ──────────────────────────────────────────────────────────────────────────────

def _cosine_lr(step: int, base_lr: float, warmup: int, total: int, min_ratio: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState) -> float:
    return _cosine_lr(
        train_state.step, base_lr,
        round(config.lr_warmup_steps),
        train_state.total_steps,
        config.lr_min_ratio,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Train state helpers
# ──────────────────────────────────────────────────────────────────────────────

def init_train_state(
    config: PretrainConfig,
    train_metadata: MRIDatasetMetadata,
    rank: int,
    world_size: int,
) -> TrainState:
    total_steps = int(
        config.epochs * train_metadata.total_slices / config.global_batch_size
    )
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank, world_size)
    return TrainState(
        step=0, total_steps=total_steps,
        model=model, optimizers=optimizers, optimizer_lrs=optimizer_lrs,
        carry=None,
    )


def _load_checkpoint(model: nn.Module, config: PretrainConfig) -> None:
    if config.load_checkpoint is None:
        return
    print(f"Loading checkpoint {config.load_checkpoint}")
    state = torch.load(config.load_checkpoint, map_location="cuda")
    model.load_state_dict(state, assign=True)


def save_train_state(config: PretrainConfig, train_state: TrainState) -> None:
    if config.checkpoint_path is None:
        return
    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(
        train_state.model.state_dict(),
        os.path.join(config.checkpoint_path, f"step_{train_state.step}"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Training step
# ──────────────────────────────────────────────────────────────────────────────

def train_batch(
    config: PretrainConfig,
    train_state: TrainState,
    batch: Dict[str, torch.Tensor],
    global_batch_size: int,
    rank: int,
    world_size: int,
) -> Optional[Dict[str, float]]:

    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return None

    batch = {k: v.cuda() for k, v in batch.items()}

    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)   # type: ignore

    train_state.carry, loss, metrics, _, _ = train_state.model(
        carry=train_state.carry, batch=batch, return_keys=[]
    )

    ((1.0 / global_batch_size) * loss).backward()

    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)
        for pg in optim.param_groups:
            pg["lr"] = lr_this_step
        optim.step()
        optim.zero_grad()

    if rank == 0 and metrics:
        assert not any(v.requires_grad for v in metrics.values())
        metric_keys   = sorted(metrics.keys())
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)
        metric_values = metric_values.cpu().numpy()
        count = max(metric_values[metric_keys.index("count")], 1)
        reduced = {}
        for i, k in enumerate(metric_keys):
            if k == "count":
                continue
            v = metric_values[i]
            # Loss fields normalised by global_batch_size; others by count
            reduced[f"train/{k}"] = v / (global_batch_size if k.endswith("loss") else count)
        reduced["train/lr"] = lr_this_step
        return reduced

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: DataLoader,
    rank: int,
    world_size: int,
) -> Optional[Dict[str, float]]:

    reduced_metrics: Optional[Dict[str, float]] = None
    metric_keys: List[str] = []
    metric_values: Optional[torch.Tensor] = None

    with torch.inference_mode():
        for set_name, batch, global_batch_size in eval_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)   # type: ignore

            # Run until all sequences halt
            while True:
                carry, loss, metrics, _, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=[]
                )
                if all_finish:
                    break

            if metric_values is None:
                metric_keys   = sorted(metrics.keys())
                metric_values = torch.zeros(len(metric_keys), dtype=torch.float32, device="cuda")

            metric_values += torch.stack([metrics[k] for k in metric_keys])

    if metric_values is not None:
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            mv = metric_values.cpu().numpy()
            count = max(mv[metric_keys.index("count")], 1)
            reduced_metrics = {}
            for i, k in enumerate(metric_keys):
                if k == "count":
                    continue
                reduced_metrics[f"eval/{k}"] = mv[i] / count

    return reduced_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Misc helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)   # type: ignore
        if config.project_name is None:
            config.project_name = "TRM-MRI"
        if config.run_name is None:
            config.run_name = f"trm_mri_{coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)
        objects = [config]
    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)
    return objects[0]   # type: ignore


def save_code_and_config(config: PretrainConfig) -> None:
    if config.checkpoint_path is None or wandb.run is None:
        return
    os.makedirs(config.checkpoint_path, exist_ok=True)
    for identifier in [config.arch.name, config.arch.loss.name]:
        path = get_model_source_path(identifier)
        if path:
            shutil.copy(path, os.path.join(config.checkpoint_path, os.path.basename(path)))
    import yaml
    with open(os.path.join(config.checkpoint_path, "all_config.yaml"), "w") as f:
        yaml.dump(config.model_dump(), f)
    wandb.run.log_code(config.checkpoint_path)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig) -> None:
    RANK       = 0
    WORLD_SIZE = 1

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK       = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)
    torch.random.manual_seed(config.seed + RANK)

    train_epochs_per_iter = config.eval_interval if config.eval_interval else config.epochs
    total_iters           = config.epochs // train_epochs_per_iter
    assert config.epochs % train_epochs_per_iter == 0

    train_loader, train_metadata = create_dataloader(
        config, "train", RANK, WORLD_SIZE,
        test_set_mode=False, epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size,
    )
    try:
        eval_loader, _ = create_dataloader(
            config, "test", RANK, WORLD_SIZE,
            test_set_mode=True, epochs_per_iter=1,
            global_batch_size=config.global_batch_size,
        )
    except Exception:
        print("NO EVAL DATA FOUND")
        eval_loader = None

    train_state = init_train_state(config, train_metadata, RANK, WORLD_SIZE)

    progress_bar = None
    ema_helper   = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(
            project=config.project_name, name=config.run_name,
            config=config.model_dump(),
            settings=wandb.Settings(_disable_stats=True),
        )
        wandb.log({"num_params": sum(p.numel() for p in train_state.model.parameters())}, step=0)
        save_code_and_config(config)

    if config.ema:
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    for _iter_id in range(total_iters):
        if RANK == 0:
            print(f"Epoch {_iter_id * train_epochs_per_iter}")

        # ── Train ──────────────────────────────────────────────────────
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size, RANK, WORLD_SIZE)
            if RANK == 0 and metrics:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)   # type: ignore
            if config.ema and ema_helper:
                ema_helper.update(train_state.model)

        if _iter_id < config.min_eval_interval:
            continue

        # ── Evaluate ───────────────────────────────────────────────────
        if eval_loader is None:
            continue

        if config.ema and ema_helper:
            train_state_eval = copy.deepcopy(train_state)
            train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
        else:
            train_state_eval = train_state
        train_state_eval.model.eval()

        metrics = evaluate(config, train_state_eval, eval_loader, RANK, WORLD_SIZE)
        if RANK == 0 and metrics:
            wandb.log(metrics, step=train_state.step)

        # ── Checkpoint ─────────────────────────────────────────────────
        if RANK == 0 and (config.checkpoint_every_eval or _iter_id == total_iters - 1):
            save_train_state(config, train_state_eval)

        if config.ema and ema_helper:
            del train_state_eval

    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
