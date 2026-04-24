from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from transformer.dataset import build_walk_forward_dataloaders
from transformer.model import PatchTSTConfig, PatchTSTSurfaceModel


@dataclass
class TrainingConfig:
    data_path: Path
    lookback: int = 252
    batch_size: int = 32
    test_fraction: float = 0.15
    val_size: int = 90
    step_size: int = 90
    min_train_size: int | None = None
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 25
    patience: int = 5
    device: str = "cuda"
    patch_len: int = 16
    stride: int = 8
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    context_dim: int = 128
    node_embed_dim: int = 32
    max_folds: int | None = None
    plot_dir: Path = PROJECT_DIR / "artifacts" / "training_plots"
    print_every_epoch: bool = True


def _move_batch_to_device(
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
) -> dict[str, torch.Tensor | list[str]]:
    moved: dict[str, torch.Tensor | list[str]] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _target_stats_to_tensors(normalizer, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    target_mean = torch.as_tensor(normalizer.target_mean, dtype=torch.float32, device=device)
    target_std = torch.as_tensor(normalizer.target_std, dtype=torch.float32, device=device)
    return target_mean, target_std


def _describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
        return f"{device} ({device_name})"
    return str(device)


def _resolve_device(device_preference: str | torch.device | None) -> torch.device:
    if isinstance(device_preference, torch.device):
        device = device_preference
    else:
        requested = (device_preference or "cuda").strip().lower()
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        torch_version = getattr(torch, "__version__", "unknown")
        cuda_runtime = getattr(torch.version, "cuda", None)
        if cuda_runtime is None:
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False. "
                f"Installed torch build: {torch_version} (CPU-only). "
                "Install a CUDA-enabled PyTorch wheel in this environment."
            )
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            f"Installed torch build: {torch_version} (CUDA runtime {cuda_runtime}). "
            "Verify your NVIDIA driver and PyTorch CUDA install match."
        )
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False.")
    return device


def _plot_fold_history(
    history: list[dict[str, float]],
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    val_mae = [entry["val_mae"] for entry in history]
    val_rmse = [entry["val_rmse"] for entry in history]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(epochs, train_loss, label="Train Loss", linewidth=2)
    axes[0].plot(epochs, val_loss, label="Val Loss", linewidth=2)
    axes[0].set_ylabel("Normalized MSE")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, val_mae, label="Val MAE", linewidth=2)
    axes[1].plot(epochs, val_rmse, label="Val RMSE", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Denormalized Error")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _print_epoch_metrics(
    fold_id: int | str,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> None:
    print(
        f"Fold {fold_id} | Epoch {epoch:03d} | "
        f"train_loss={train_metrics['loss']:.6f} | "
        f"val_loss={val_metrics['loss']:.6f} | "
        f"val_mae={val_metrics['mae']:.6f} | "
        f"val_rmse={val_metrics['rmse']:.6f}"
    )


def build_model_from_batch(
    batch: dict[str, torch.Tensor | list[str]],
    config: TrainingConfig,
) -> PatchTSTSurfaceModel:
    surface_history = batch["surface_history"]
    rate_history = batch["rate_history"]
    feature_history = batch["feature_history"]

    if not isinstance(surface_history, torch.Tensor):
        raise TypeError("surface_history must be a torch.Tensor")
    if not isinstance(rate_history, torch.Tensor):
        raise TypeError("rate_history must be a torch.Tensor")
    if not isinstance(feature_history, torch.Tensor):
        raise TypeError("feature_history must be a torch.Tensor")

    lookback = surface_history.shape[1]
    if lookback != config.lookback:
        raise ValueError(f"Expected lookback {config.lookback}, got {lookback}")

    model_config = PatchTSTConfig(
        lookback=lookback,
        num_maturities=surface_history.shape[2],
        num_deltas=surface_history.shape[3],
        num_surface_nodes=surface_history.shape[2] * surface_history.shape[3],
        num_rate_features=rate_history.shape[2],
        num_scalar_features=feature_history.shape[2],
        patch_len=config.patch_len,
        stride=config.stride,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        context_dim=config.context_dim,
        node_embed_dim=config.node_embed_dim,
    )
    return PatchTSTSurfaceModel(model_config)


def train_one_epoch(
    model: PatchTSTSurfaceModel,
    loader,
    optimizer: AdamW,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        surface_history = batch["surface_history"]
        rate_history = batch["rate_history"]
        feature_history = batch["feature_history"]
        target_surface_delta = batch["target_surface_delta"]

        optimizer.zero_grad(set_to_none=True)
        pred_surface_delta = model(surface_history, rate_history, feature_history)
        loss = loss_fn(pred_surface_delta, target_surface_delta)
        loss.backward()
        optimizer.step()

        batch_size = surface_history.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
    }


def validate_one_epoch(
    model: PatchTSTSurfaceModel,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
    normalizer,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_abs_error = 0.0
    total_squared_error = 0.0
    total_elements = 0

    target_mean, target_std = _target_stats_to_tensors(normalizer, device)

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            surface_history = batch["surface_history"]
            rate_history = batch["rate_history"]
            feature_history = batch["feature_history"]
            target_surface_delta = batch["target_surface_delta"]

            pred_surface_delta = model(surface_history, rate_history, feature_history)
            loss = loss_fn(pred_surface_delta, target_surface_delta)

            pred_raw = pred_surface_delta * target_std + target_mean
            target_raw = target_surface_delta * target_std + target_mean
            diff = pred_raw - target_raw

            batch_size = surface_history.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            total_abs_error += diff.abs().sum().item()
            total_squared_error += diff.square().sum().item()
            total_elements += diff.numel()

    return {
        "loss": total_loss / max(total_samples, 1),
        "mae": total_abs_error / max(total_elements, 1),
        "rmse": (total_squared_error / max(total_elements, 1)) ** 0.5,
    }


def run_walk_forward_validation(config: TrainingConfig) -> dict[str, object]:
    loader_bundle = build_walk_forward_dataloaders(
        data_path=config.data_path,
        lookback=config.lookback,
        batch_size=config.batch_size,
        test_fraction=config.test_fraction,
        val_size=config.val_size,
        step_size=config.step_size,
        min_train_size=config.min_train_size,
    )
    device = _resolve_device(config.device)
    loss_fn = nn.MSELoss()

    print(f"Using device: {_describe_device(device)}")

    fold_loaders = loader_bundle.folds
    if config.max_folds is not None:
        fold_loaders = fold_loaders[: config.max_folds]

    fold_summaries: list[dict[str, object]] = []

    for fold in fold_loaders:
        sample_batch = next(iter(fold.train))
        model = build_model_from_batch(sample_batch, config).to(device)
        optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        best_val_loss = float("inf")
        best_epoch = 0
        best_metrics: dict[str, float] | None = None
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        epoch_history: list[dict[str, float]] = []

        for epoch in range(1, config.num_epochs + 1):
            train_metrics = train_one_epoch(model, fold.train, optimizer, loss_fn, device)
            val_metrics = validate_one_epoch(
                model,
                fold.val,
                loss_fn,
                device,
                fold.normalizer,
            )
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
            }
            epoch_history.append(epoch_record)

            if config.print_every_epoch:
                _print_epoch_metrics(fold.fold_id, epoch, train_metrics, val_metrics)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                best_metrics = {
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "val_mae": val_metrics["mae"],
                    "val_rmse": val_metrics["rmse"],
                }
                best_state = deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        if best_metrics is None:
            raise RuntimeError(f"Fold {fold.fold_id} did not produce validation metrics")

        plot_path = config.plot_dir / f"fold_{fold.fold_id:02d}_metrics.png"
        _plot_fold_history(
            epoch_history,
            plot_path,
            title=f"Fold {fold.fold_id} Training Curves",
        )

        fold_summary = {
            "fold_id": fold.fold_id,
            "best_epoch": best_epoch,
            "train_target_start": fold.train_target_start,
            "train_target_end": fold.train_target_end,
            "val_target_start": fold.val_target_start,
            "val_target_end": fold.val_target_end,
            "plot_path": str(plot_path),
            "history": epoch_history,
            **best_metrics,
        }
        fold_summaries.append(fold_summary)

        print(
            f"Fold {fold.fold_id} complete | "
            f"best_epoch={best_epoch} | "
            f"best_val_loss={best_metrics['val_loss']:.6f} | "
            f"best_val_mae={best_metrics['val_mae']:.6f} | "
            f"best_val_rmse={best_metrics['val_rmse']:.6f} | "
            f"plot={plot_path}"
        )

    val_losses = np.array([fold["val_loss"] for fold in fold_summaries], dtype=float)
    val_maes = np.array([fold["val_mae"] for fold in fold_summaries], dtype=float)
    val_rmses = np.array([fold["val_rmse"] for fold in fold_summaries], dtype=float)

    aggregate = {
        "num_folds": len(fold_summaries),
        "val_loss_mean": float(val_losses.mean()),
        "val_loss_std": float(val_losses.std()),
        "val_mae_mean": float(val_maes.mean()),
        "val_mae_std": float(val_maes.std()),
        "val_rmse_mean": float(val_rmses.mean()),
        "val_rmse_std": float(val_rmses.std()),
    }

    return {
        "folds": fold_summaries,
        "aggregate": aggregate,
    }


def train_final_and_test(config: TrainingConfig) -> dict[str, object]:
    loader_bundle = build_walk_forward_dataloaders(
        data_path=config.data_path,
        lookback=config.lookback,
        batch_size=config.batch_size,
        test_fraction=config.test_fraction,
        val_size=config.val_size,
        step_size=config.step_size,
        min_train_size=config.min_train_size,
    )
    device = _resolve_device(config.device)
    loss_fn = nn.MSELoss()

    print(f"Using device: {_describe_device(device)}")

    sample_batch = next(iter(loader_bundle.final_train))
    model = build_model_from_batch(sample_batch, config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    train_history: list[dict[str, float]] = []
    for epoch in range(1, config.num_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            loader_bundle.final_train,
            optimizer,
            loss_fn,
            device,
        )
        train_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
        }
        train_history.append(train_record)
        if config.print_every_epoch:
            print(
                f"Final train | Epoch {epoch:03d} | "
                f"train_loss={train_metrics['loss']:.6f}"
            )

    test_metrics = validate_one_epoch(
        model,
        loader_bundle.test,
        loss_fn,
        device,
        loader_bundle.final_normalizer,
    )

    final_plot_path = config.plot_dir / "final_train_loss.png"
    final_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        [entry["epoch"] for entry in train_history],
        [entry["train_loss"] for entry in train_history],
        linewidth=2,
    )
    ax.set_title("Final Train Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized MSE")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(final_plot_path, dpi=150)
    plt.close(fig)

    print(
        f"Final test | "
        f"loss={test_metrics['loss']:.6f} | "
        f"mae={test_metrics['mae']:.6f} | "
        f"rmse={test_metrics['rmse']:.6f} | "
        f"plot={final_plot_path}"
    )

    return {
        "final_train_history": train_history,
        "test_metrics": test_metrics,
        "plot_path": str(final_plot_path),
    }


if __name__ == "__main__":
    _resolve_device("cuda")
    config = TrainingConfig(
        data_path=PROJECT_DIR / "data" / "raw" / "all_data.xlsx",
    )

    validation_results = run_walk_forward_validation(config)
    print("Walk-forward validation aggregate:", validation_results["aggregate"])
    print("First fold summary:", validation_results["folds"][0])
