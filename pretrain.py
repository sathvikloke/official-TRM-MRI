"""
pretrain.py  (MRI version)

Modified from the Samsung TRM pretrain.py to work with MRI k-space data.

Changes from original:
    1. PuzzleDataset replaced with MRIDataset -- reads .npy files from
       dataset/build_mri_dataset.py
    2. model_cfg passes height and width from dataset.json so
       TinyRecursiveReasoningModel_MRIConfig can instantiate correctly
    3. _iter_test fixed (Bug #5): all ranks skip the last partial batch
       together to prevent distributed allreduce deadlock
    4. scales field included in dataloader batches (needed if evaluators
       want to report un-normalised PSNR in physical units)
    5. Everything else unchanged: distributed training, EMA, W&B,
       checkpointing, cosine LR, AdamATan2, gradient allreduce
"""

from typing import Optional, Any, Sequence, List, Dict
from dataclasses import dataclass
import os
import json
import math
import yaml
import shutil
import copy

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2 import AdamATan2

from utils.functions import load_model_class, get_model_source_path
from models.ema import EMAHelper


# ---------------------------------------------------------------------------
# MRI dataset metadata  (mirrors MRIDatasetMetadata in build_mri_dataset.py)
# ---------------------------------------------------------------------------

class MRIDatasetMetadata(pydantic.BaseModel):
    height:          int
    width:           int
    seq_len:         int
    num_slices:      int
    acceleration:    int
    center_fraction: float
    is_multicoil:    bool
    num_files:       int
    sets:            List[str] = ["all"]


# ---------------------------------------------------------------------------
# MRI dataloader
# ---------------------------------------------------------------------------

class MRIDatasetConfig(pydantic.BaseModel):
    dataset_path:      str
    global_batch_size: int
    rank:              int
    num_replicas:      int
    seed:              int  = 0
    test_set_mode:     bool = False
    epochs_per_iter:   int  = 1


class MRIDataset(IterableDataset):
    """
    Yields batches of:
        inputs : (local_B, 2, H*W)  float32  normalised masked k-space
        labels : (local_B, H*W)     float32  normalised RSS image
        masks  : (local_B, W)       float32  undersampling mask
        scales : (local_B,)         float32  per-slice normalisation factor
    """

    def __init__(self, config: MRIDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split  = split

        split_dir = os.path.join(config.dataset_path, split)
        with open(os.path.join(split_dir, "dataset.json"), "r") as f:
            self.metadata = MRIDatasetMetadata(**json.load(f))

        assert config.global_batch_size % config.num_replicas == 0, (
            f"global_batch_size {config.global_batch_size} must be divisible "
            f"by world_size {config.num_replicas}"
        )
        self.local_batch_size = config.global_batch_size // config.num_replicas

        self._inputs = None
        self._labels = None
        self._masks  = None
        self._scales = None
        self._epoch  = 0

    def _lazy_load(self):
        if self._inputs is not None:
            return
        d = os.path.join(self.config.dataset_path, self.split)
        self._inputs = np.load(os.path.join(d, "all__inputs.npy"), mmap_mode="r")
        self._labels = np.load(os.path.join(d, "all__labels.npy"), mmap_mode="r")
        self._masks  = np.load(os.path.join(d, "all__masks.npy"),  mmap_mode="r")
        self._scales = np.load(os.path.join(d, "all__scales.npy"), mmap_mode="r")

    def _make_batch(self, indices: np.ndarray) -> Dict[str, torch.Tensor]:
        return {
            "inputs": torch.from_numpy(self._inputs[indices].copy()),
            "labels": torch.from_numpy(self._labels[indices].copy()),
            "masks":  torch.from_numpy(self._masks[indices].copy()),
            "scales": torch.from_numpy(self._scales[indices].copy()),
        }

    def _iter_train(self):
        N   = self._inputs.shape[0]
        rng = np.random.default_rng(self.config.seed + self._epoch)
        self._epoch += 1

        indices = np.concatenate(
            [rng.permutation(N) for _ in range(self.config.epochs_per_iter)]
        )

        start = 0
        while start + self.config.global_batch_size <= len(indices):
            global_idx = indices[start: start + self.config.global_batch_size]
            lo = self.config.rank * self.local_batch_size
            hi = lo + self.local_batch_size
            yield "all", self._make_batch(global_idx[lo:hi]), self.config.global_batch_size
            start += self.config.global_batch_size

    def _iter_test(self):
        """
        Bug #5 fix: all ranks skip the last incomplete batch together.
        In the original code, some ranks could get an empty local slice
        while others did not, causing a distributed allreduce deadlock.
        Now we only yield batches where the full global batch is available.
        """
        N     = self._inputs.shape[0]
        start = 0
        while start + self.config.global_batch_size <= N:
            # Only yield when we have a FULL global batch
            end = start + self.config.global_batch_size
            lo  = start + self.config.rank       * self.local_batch_size
            hi  = start + (self.config.rank + 1) * self.local_batch_size
            yield "all", self._make_batch(np.arange(lo, hi)), end - start
            start += self.config.global_batch_size
        # Last partial batch is silently dropped -- avoids deadlock

    def __iter__(self):
        self._lazy_load()
        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()


# ---------------------------------------------------------------------------
# Pretrain config
# ---------------------------------------------------------------------------

class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    arch:            ArchConfig
    data_paths:      List[str]
    data_paths_test: List[str] = []
    evaluators:      List[EvaluatorConfig] = []

    global_batch_size: int
    epochs:            int

    lr:              float
    lr_min_ratio:    float = 1.0
    lr_warmup_steps: int   = 2000

    weight_decay: float = 0.1
    beta1:        float = 0.9
    beta2:        float = 0.95

    # Kept for YAML compat -- not used in MRI training
    puzzle_emb_lr:           float = 1e-2
    puzzle_emb_weight_decay: float = 0.1

    project_name:    Optional[str] = None
    run_name:        Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    seed:                  int        = 0
    checkpoint_every_eval: bool       = False
    eval_interval:         Optional[int] = None
    min_eval_interval:     Optional[int] = 0
    eval_save_outputs:     List[str]  = []

    ema:            bool  = False
    ema_rate:       float = 0.999
    freeze_weights: bool  = False


@dataclass
class TrainState:
    model:         nn.Module
    optimizers:    Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry:         Any
    step:          int
    total_steps:   int


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------

def create_dataloader(config:     PretrainConfig,
                      split:      str,
                      rank:       int,
                      world_size: int,
                      **kwargs):
    dataset_path = (
        config.data_paths_test[0]
        if len(config.data_paths_test) > 0 and split == "test"
        else config.data_paths[0]
    )

    dataset = MRIDataset(
        MRIDatasetConfig(
            dataset_path=dataset_path,
            global_batch_size=config.global_batch_size,
            rank=rank,
            num_replicas=world_size,
            seed=config.seed,
            **kwargs,
        ),
        split=split,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True,
    )

    return dataloader, dataset.metadata


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_model(config:         PretrainConfig,
                 train_metadata: MRIDatasetMetadata,
                 rank:           int,
                 world_size:     int):
    """
    Build model config dict and instantiate model + loss head.
    height and width come from dataset.json so the model knows spatial dims.
    seq_len is also passed for interface compat (ignored internally by trm_mri).
    """
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size // world_size,
        height=train_metadata.height,
        width=train_metadata.width,
        seq_len=train_metadata.seq_len,         # compat field in trm_mri config
        vocab_size=1,
        num_puzzle_identifiers=1,
        causal=False,
    )

    model_cls     = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)

        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)

        if rank == 0:
            load_checkpoint(model, config)

        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    optimizers = [
        AdamATan2(
            model.parameters(),
            lr=0,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )
    ]
    optimizer_lrs = [config.lr]

    return model, optimizers, optimizer_lrs


# ---------------------------------------------------------------------------
# LR schedule  (unchanged from original)
# ---------------------------------------------------------------------------

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    base_lr:            float,
    num_warmup_steps:   int,
    num_training_steps: int,
    min_ratio:          float = 0.0,
    num_cycles:         float = 0.5,
) -> float:
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return base_lr * (
        min_ratio
        + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    )


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState) -> float:
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio,
    )


# ---------------------------------------------------------------------------
# Train state
# ---------------------------------------------------------------------------

def init_train_state(config:         PretrainConfig,
                     train_metadata: MRIDatasetMetadata,
                     rank:           int,
                     world_size:     int) -> TrainState:

    total_steps = int(
        config.epochs * train_metadata.num_slices / config.global_batch_size
    )

    model, optimizers, optimizer_lrs = create_model(
        config, train_metadata, rank=rank, world_size=world_size
    )

    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    if config.checkpoint_path is None:
        return
    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(
        train_state.model.state_dict(),
        os.path.join(config.checkpoint_path, f"step_{train_state.step}"),
    )


def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")
        state_dict = torch.load(config.load_checkpoint, map_location="cuda")
        model.load_state_dict(state_dict, assign=True)


# ---------------------------------------------------------------------------
# Train batch  (unchanged from original)
# ---------------------------------------------------------------------------

def train_batch(config:            PretrainConfig,
                train_state:       TrainState,
                batch:             Any,
                global_batch_size: int,
                rank:              int,
                world_size:        int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:
        return

    batch = {k: v.cuda() for k, v in batch.items()}

    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)

    train_state.carry, loss, metrics, _, _ = train_state.model(
        carry=train_state.carry, batch=batch, return_keys=[]
    )

    ((1 / global_batch_size) * loss).backward()

    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)

    lr_this_step = None
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)
        for param_group in optim.param_groups:
            param_group["lr"] = lr_this_step
        optim.step()
        optim.zero_grad()

    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())
        metric_keys   = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k] for k in metric_keys])

        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            count = max(float(metric_values[metric_keys.index("count")]), 1.0)
            reduced = {}
            for i, k in enumerate(metric_keys):
                if k == "count":
                    continue
                elif k.endswith("loss"):
                    reduced[f"train/{k}"] = metric_values[i] / global_batch_size
                else:
                    reduced[f"train/{k}"] = metric_values[i] / count
            reduced["train/lr"] = lr_this_step
            return reduced


# ---------------------------------------------------------------------------
# Evaluate  (unchanged from original)
# ---------------------------------------------------------------------------

def evaluate(config:        PretrainConfig,
             train_state:   TrainState,
             eval_loader,
             eval_metadata: MRIDatasetMetadata,
             evaluators:    List[Any],
             rank:          int,
             world_size:    int,
             cpu_group:     Optional[dist.ProcessGroup]):

    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        metric_keys   = []
        metric_values = None
        save_preds    = {}

        processed_batches = 0
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")

            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)

            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1
                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            if metric_values is None:
                metric_keys   = list(sorted(metrics.keys()))
                metric_values = torch.zeros(
                    len(metrics), dtype=torch.float32, device="cuda"
                )

            metric_values += torch.stack([metrics[k] for k in metric_keys])

        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                mv    = metric_values.cpu().numpy()
                count = max(float(mv[metric_keys.index("count")]), 1.0)
                reduced_metrics = {}
                for i, k in enumerate(metric_keys):
                    if k == "count":
                        continue
                    elif k.endswith("loss"):
                        reduced_metrics[f"eval/{k}"] = mv[i]
                    else:
                        reduced_metrics[f"eval/{k}"] = mv[i] / count

        for evaluator in evaluators:
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            metrics_ev = evaluator.result(
                evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group
            )
            if rank == 0 and metrics_ev is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}
                reduced_metrics.update(metrics_ev)

    return reduced_metrics


# ---------------------------------------------------------------------------
# Config helpers  (unchanged)
# ---------------------------------------------------------------------------

def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return
    os.makedirs(config.checkpoint_path, exist_ok=True)
    for code_file in [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name),
    ]:
        if code_file is not None:
            shutil.copy(
                code_file,
                os.path.join(config.checkpoint_path, os.path.basename(code_file)),
            )
    with open(os.path.join(config.checkpoint_path, "all_config.yaml"), "wt") as f:
        yaml.dump(config.model_dump(), f)
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig,
                       rank:         int,
                       world_size:   int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)
        if config.project_name is None:
            config.project_name = "TRM-MRI"
        if config.run_name is None:
            config.run_name = (
                f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
            )
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join(
                "checkpoints", config.project_name, config.run_name
            )
        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]


# ---------------------------------------------------------------------------
# Main training loop  (unchanged from original)
# ---------------------------------------------------------------------------

@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK              = 0
    WORLD_SIZE        = 1
    CPU_PROCESS_GROUP = None

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK       = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")

    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)
    torch.random.manual_seed(config.seed + RANK)

    train_epochs_per_iter = (
        config.eval_interval if config.eval_interval is not None else config.epochs
    )
    total_iters = config.epochs // train_epochs_per_iter
    assert config.epochs % train_epochs_per_iter == 0, (
        "eval_interval must divide epochs evenly"
    )

    train_loader, train_metadata = create_dataloader(
        config, "train",
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        rank=RANK,
        world_size=WORLD_SIZE,
    )

    try:
        eval_loader, eval_metadata = create_dataloader(
            config, "test",
            test_set_mode=True,
            epochs_per_iter=1,
            rank=RANK,
            world_size=WORLD_SIZE,
        )
    except Exception:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    evaluators = []

    train_state = init_train_state(
        config, train_metadata, rank=RANK, world_size=WORLD_SIZE
    )

    progress_bar = None
    ema_helper   = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=config.model_dump(),
            settings=wandb.Settings(_disable_stats=True),
        )
        wandb.log(
            {"num_params": sum(x.numel() for x in train_state.model.parameters())},
            step=0,
        )
        save_code_and_config(config)

    if config.ema:
        print("Setup EMA")
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    for _iter_id in range(total_iters):
        print(f"[Rank {RANK}, World {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        # ── Train ─────────────────────────────────────────────────────────
        if RANK == 0:
            print("TRAIN")
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(
                config, train_state, batch, global_batch_size,
                rank=RANK, world_size=WORLD_SIZE,
            )
            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)
            if config.ema:
                ema_helper.update(train_state.model)

        if _iter_id >= config.min_eval_interval:
            # ── Evaluate ──────────────────────────────────────────────────
            if RANK == 0:
                print("EVALUATE")
            if config.ema:
                train_state_eval       = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state

            train_state_eval.model.eval()
            metrics = evaluate(
                config, train_state_eval,
                eval_loader, eval_metadata, evaluators,
                rank=RANK, world_size=WORLD_SIZE,
                cpu_group=CPU_PROCESS_GROUP,
            )

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)

            if RANK == 0 and (
                config.checkpoint_every_eval or (_iter_id == total_iters - 1)
            ):
                save_train_state(config, train_state_eval)

            if config.ema:
                del train_state_eval

    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
