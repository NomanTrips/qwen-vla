"""Training script for QwenVLA."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.util
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from models import QwenVLA, QwenVLAConfig
from training.loss import LossWeighting, flow_matching_loss
from training.scheduler import build_cosine_schedule_with_warmup, get_trainable_parameters


@dataclass
class DatasetConfig:
    module: str
    class_name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    collate_module: str | None = None
    collate_fn: str | None = None


@dataclass
class TrainingConfig:
    batch_size: int = 8
    num_epochs: int = 1
    max_steps: int = 0
    grad_accumulation: int = 1
    learning_rate: float = 1e-4
    projector_learning_rate: float | None = None
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    log_every: int = 10
    save_every: int = 1000
    output_dir: str = "checkpoints"
    resume_from: str | None = None
    mixed_precision: str = "bf16"
    loss_weighting: LossWeighting = "none"
    beta_a: float = 1.0
    beta_b: float = 1.0
    use_wandb: bool = False
    wandb_project: str = "qwen-vla"
    wandb_run_name: str | None = None
    num_workers: int = 4


@dataclass
class Config:
    model: QwenVLAConfig = field(default_factory=QwenVLAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig | None = None


def _update_dataclass(instance: Any, updates: Mapping[str, Any]) -> Any:
    for key, value in updates.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if dataclasses.is_dataclass(current) and isinstance(value, Mapping):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: str | None) -> Config:
    config = Config()
    if not path:
        return config
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if "model" in payload:
        _update_dataclass(config.model, payload["model"])
    if "training" in payload:
        _update_dataclass(config.training, payload["training"])
    if "dataset" in payload and payload["dataset"]:
        dataset_payload = payload["dataset"]
        config.dataset = DatasetConfig(
            module=dataset_payload["module"],
            class_name=dataset_payload["class_name"],
            kwargs=dataset_payload.get("kwargs", {}),
            collate_module=dataset_payload.get("collate_module"),
            collate_fn=dataset_payload.get("collate_fn"),
        )
    return config


def _resolve_collate_fn(dataset_config: DatasetConfig | None):
    if not dataset_config or not dataset_config.collate_module or not dataset_config.collate_fn:
        return None
    module = importlib.import_module(dataset_config.collate_module)
    return getattr(module, dataset_config.collate_fn)


def build_dataloader(dataset_config: DatasetConfig, training_config: TrainingConfig) -> DataLoader:
    module = importlib.import_module(dataset_config.module)
    dataset_cls = getattr(module, dataset_config.class_name)
    dataset = dataset_cls(**dataset_config.kwargs)
    collate_fn = _resolve_collate_fn(dataset_config)
    return DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )


def _is_weight_decay_exempt(name: str, param: nn.Parameter) -> bool:
    if not param.requires_grad:
        return True
    if name.endswith(".bias"):
        return True
    if "norm" in name.lower() or "layernorm" in name.lower():
        return True
    return False


def build_optimizer(model: QwenVLA, config: TrainingConfig) -> AdamW:
    decay_params = []
    no_decay_params = []
    projector_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("feature_projector") or name.startswith("state_projector"):
            projector_params.append((name, param))
            continue
        if _is_weight_decay_exempt(name, param):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append(
            {"params": decay_params, "weight_decay": config.weight_decay, "lr": config.learning_rate}
        )
    if no_decay_params:
        param_groups.append(
            {"params": no_decay_params, "weight_decay": 0.0, "lr": config.learning_rate}
        )
    if projector_params:
        proj_decay = []
        proj_no_decay = []
        for name, param in projector_params:
            if _is_weight_decay_exempt(name, param):
                proj_no_decay.append(param)
            else:
                proj_decay.append(param)
        projector_lr = config.projector_learning_rate or config.learning_rate
        if proj_decay:
            param_groups.append(
                {"params": proj_decay, "weight_decay": config.weight_decay, "lr": projector_lr}
            )
        if proj_no_decay:
            param_groups.append(
                {"params": proj_no_decay, "weight_decay": 0.0, "lr": projector_lr}
            )

    return AdamW(param_groups, lr=config.learning_rate, betas=(0.9, 0.999))


def _maybe_init_wandb(config: TrainingConfig):
    if not config.use_wandb:
        return None
    if importlib.util.find_spec("wandb") is None:
        raise RuntimeError("wandb is enabled but not installed.")
    import wandb

    return wandb.init(project=config.wandb_project, name=config.wandb_run_name)


def _extract_batch(batch: Any) -> Dict[str, Any]:
    if isinstance(batch, Mapping):
        return {
            "images": batch["images"],
            "text": batch.get("text") or batch.get("instruction") or "",
            "states": batch.get("states"),
            "actions": batch.get("actions") or batch.get("action_chunk"),
        }
    raise ValueError("Batch must be a mapping with images and actions.")


def _build_autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "none":
        return torch.autocast(device_type=device.type, enabled=False)
    if precision == "fp16":
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    raise ValueError(f"Unknown mixed precision setting: {precision}")


def _load_checkpoint(path: str, model: nn.Module, optimizer, scheduler, scaler) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def train(config_path: str | None) -> None:
    config = load_config(config_path)
    if config.dataset is None:
        raise ValueError("Dataset configuration is required to start training.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = QwenVLA(config.model)
    optimizer = build_optimizer(model, config.training)

    dataloader = build_dataloader(config.dataset, config.training)
    total_steps = config.training.max_steps
    if total_steps <= 0:
        total_steps = config.training.num_epochs * len(dataloader)

    scheduler = build_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = None
    if config.training.mixed_precision == "fp16" and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")

    if config.training.resume_from:
        checkpoint = _load_checkpoint(
            config.training.resume_from, model, optimizer, scheduler, scaler
        )
        start_epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("step", 0)
        best_loss = checkpoint.get("best_loss", best_loss)

    os.makedirs(config.training.output_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(config.training.output_dir, "logs"))
    wandb_run = _maybe_init_wandb(config.training)

    model.train()

    for epoch in range(start_epoch, config.training.num_epochs):
        for batch in dataloader:
            batch_data = _extract_batch(batch)
            actions = batch_data["actions"].to(device)

            autocast_context = _build_autocast_context(device, config.training.mixed_precision)
            with autocast_context:
                loss = flow_matching_loss(
                    model,
                    batch_data["images"],
                    batch_data["text"],
                    batch_data["states"],
                    actions,
                    beta_a=config.training.beta_a,
                    beta_b=config.training.beta_b,
                    loss_weighting=config.training.loss_weighting,
                )
                loss = loss / config.training.grad_accumulation

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (global_step + 1) % config.training.grad_accumulation == 0:
                if config.training.max_grad_norm > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        get_trainable_parameters(model.parameters()),
                        config.training.max_grad_norm,
                    )

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            if global_step % config.training.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)
                if wandb_run is not None:
                    wandb_run.log({"loss": loss.item(), "lr": lr}, step=global_step)

            if (global_step + 1) % config.training.save_every == 0:
                ckpt_path = os.path.join(
                    config.training.output_dir, f"checkpoint_step_{global_step + 1}.pt"
                )
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "step": global_step + 1,
                        "epoch": epoch,
                        "best_loss": best_loss,
                    },
                    ckpt_path,
                )

            current_loss = loss.item() * config.training.grad_accumulation
            if current_loss < best_loss:
                best_loss = current_loss
                best_path = os.path.join(config.training.output_dir, "best.pt")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "step": global_step + 1,
                        "epoch": epoch,
                        "best_loss": best_loss,
                    },
                    best_path,
                )

            global_step += 1
            if config.training.max_steps and global_step >= config.training.max_steps:
                break
        if config.training.max_steps and global_step >= config.training.max_steps:
            break

    writer.close()
    if wandb_run is not None:
        wandb_run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QwenVLA")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
