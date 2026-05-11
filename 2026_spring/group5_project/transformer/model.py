from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from torch import nn


@dataclass
class PatchTSTConfig:
    lookback: int = 120
    num_maturities: int = 10
    num_deltas: int = 17
    num_surface_nodes: int = 170
    num_rate_features: int = 10
    num_scalar_features: int = 9
    patch_len: int = 16
    stride: int = 8
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    context_dim: int = 128
    node_embed_dim: int = 32

    def __post_init__(self) -> None:
        if self.lookback < 1:
            raise ValueError("lookback must be positive")
        if self.patch_len < 1:
            raise ValueError("patch_len must be positive")
        if self.stride < 1:
            raise ValueError("stride must be positive")
        if self.patch_len > self.lookback:
            raise ValueError("patch_len must be less than or equal to lookback")
        if self.num_surface_nodes != self.num_maturities * self.num_deltas:
            raise ValueError("num_surface_nodes must equal num_maturities * num_deltas")

    @property
    def num_patches(self) -> int:
        return (self.lookback - self.patch_len) // self.stride + 1

    @property
    def context_input_dim(self) -> int:
        return self.lookback * (self.num_rate_features + self.num_scalar_features)

    @property
    def per_node_repr_dim(self) -> int:
        return self.num_patches * self.d_model

    @property
    def fused_dim(self) -> int:
        return self.per_node_repr_dim + self.context_dim + self.node_embed_dim


class PatchTSTSurfaceModel(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.config = config

        self.patch_proj = nn.Linear(config.patch_len, config.d_model)
        self.patch_pos_embedding = nn.Parameter(
            torch.zeros(1, config.num_patches, config.d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        self.context_mlp = nn.Sequential(
            nn.LayerNorm(config.context_input_dim),
            nn.Linear(config.context_input_dim, config.context_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.context_dim, config.context_dim),
            nn.GELU(),
        )

        self.node_embedding = nn.Embedding(
            config.num_surface_nodes,
            config.node_embed_dim,
        )
        self.register_buffer(
            "node_indices",
            torch.arange(config.num_surface_nodes, dtype=torch.long),
            persistent=False,
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(config.fused_dim),
            nn.Linear(config.fused_dim, 256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 1),
        )

        nn.init.trunc_normal_(self.patch_pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.node_embedding.weight, std=0.02)

    def _validate_inputs(
        self,
        surface_history: torch.Tensor,
        rate_history: torch.Tensor,
        feature_history: torch.Tensor,
    ) -> None:
        if surface_history.ndim != 4:
            raise ValueError("surface_history must have shape [B, L, M, D]")
        if rate_history.ndim != 3:
            raise ValueError("rate_history must have shape [B, L, M]")
        if feature_history.ndim != 3:
            raise ValueError("feature_history must have shape [B, L, F]")

        batch_size, lookback, num_maturities, num_deltas = surface_history.shape
        if lookback != self.config.lookback:
            raise ValueError(
                f"Expected lookback={self.config.lookback}, got {lookback}"
            )
        if num_maturities != self.config.num_maturities:
            raise ValueError(
                f"Expected num_maturities={self.config.num_maturities}, got {num_maturities}"
            )
        if num_deltas != self.config.num_deltas:
            raise ValueError(
                f"Expected num_deltas={self.config.num_deltas}, got {num_deltas}"
            )

        if rate_history.shape[0] != batch_size or rate_history.shape[1] != lookback:
            raise ValueError("rate_history batch/sequence dimensions must match surface_history")
        if rate_history.shape[2] != self.config.num_rate_features:
            raise ValueError(
                f"Expected num_rate_features={self.config.num_rate_features}, got {rate_history.shape[2]}"
            )

        if feature_history.shape[0] != batch_size or feature_history.shape[1] != lookback:
            raise ValueError("feature_history batch/sequence dimensions must match surface_history")
        if feature_history.shape[2] != self.config.num_scalar_features:
            raise ValueError(
                f"Expected num_scalar_features={self.config.num_scalar_features}, got {feature_history.shape[2]}"
            )

    def _flatten_surface_to_channels(self, surface_history: torch.Tensor) -> torch.Tensor:
        batch_size, lookback, _, _ = surface_history.shape
        return (
            surface_history.reshape(batch_size, lookback, self.config.num_surface_nodes)
            .transpose(1, 2)
            .contiguous()
        )

    def _extract_patches(self, surface_channels: torch.Tensor) -> torch.Tensor:
        patches = surface_channels.unfold(
            dimension=-1,
            size=self.config.patch_len,
            step=self.config.stride,
        )
        if patches.shape[-2] != self.config.num_patches:
            raise ValueError(
                f"Expected {self.config.num_patches} patches, got {patches.shape[-2]}"
            )
        return patches

    def _reshape_output(self, pred_flat: torch.Tensor) -> torch.Tensor:
        batch_size = pred_flat.shape[0]
        return pred_flat.reshape(
            batch_size,
            self.config.num_maturities,
            self.config.num_deltas,
        )

    def forward(
        self,
        surface_history: torch.Tensor,
        rate_history: torch.Tensor,
        feature_history: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(surface_history, rate_history, feature_history)

        surface_channels = self._flatten_surface_to_channels(surface_history)
        patches = self._extract_patches(surface_channels)
        batch_size, num_nodes, num_patches, _ = patches.shape

        patch_tokens = self.patch_proj(patches)
        patch_tokens = patch_tokens.reshape(
            batch_size * num_nodes,
            num_patches,
            self.config.d_model,
        )
        patch_tokens = patch_tokens + self.patch_pos_embedding
        encoded = self.temporal_encoder(patch_tokens)
        encoded = encoded.reshape(
            batch_size,
            num_nodes,
            num_patches,
            self.config.d_model,
        )
        surface_repr = encoded.reshape(batch_size, num_nodes, self.config.per_node_repr_dim)

        context_input = torch.cat(
            [
                rate_history.reshape(batch_size, -1),
                feature_history.reshape(batch_size, -1),
            ],
            dim=-1,
        )
        context_repr = self.context_mlp(context_input)
        context_repr = context_repr.unsqueeze(1).expand(-1, num_nodes, -1)

        node_repr = self.node_embedding(self.node_indices)
        node_repr = node_repr.unsqueeze(0).expand(batch_size, -1, -1)

        fused_repr = torch.cat([surface_repr, context_repr, node_repr], dim=-1)
        pred_flat = self.output_head(fused_repr).squeeze(-1)
        return self._reshape_output(pred_flat)


if __name__ == "__main__":
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))

    from transformer.dataset import build_walk_forward_dataloaders

    loader_bundle = build_walk_forward_dataloaders(
        data_path=PROJECT_DIR / "data" / "raw" / "all_data.xlsx",
        batch_size=4,
    )
    batch = next(iter(loader_bundle.folds[0].train))
    config = PatchTSTConfig(
        lookback=batch["surface_history"].shape[1],
        num_maturities=batch["surface_history"].shape[2],
        num_deltas=batch["surface_history"].shape[3],
        num_surface_nodes=batch["surface_history"].shape[2] * batch["surface_history"].shape[3],
        num_rate_features=batch["rate_history"].shape[2],
        num_scalar_features=batch["feature_history"].shape[2],
    )
    model = PatchTSTSurfaceModel(config)
    output = model(
        batch["surface_history"],
        batch["rate_history"],
        batch["feature_history"],
    )
    print(f"surface_history shape: {batch['surface_history'].shape}")
    print(f"rate_history shape: {batch['rate_history'].shape}")
    print(f"feature_history shape: {batch['feature_history'].shape}")
    print(f"output shape: {output.shape}")
