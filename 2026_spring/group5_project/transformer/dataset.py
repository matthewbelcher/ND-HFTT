from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


EPSILON = 1e-6


@dataclass
class DailySurfaceData:
    dates: np.ndarray
    maturities: np.ndarray
    deltas: np.ndarray
    surfaces: np.ndarray
    rate_curves: np.ndarray
    daily_features: np.ndarray
    feature_names: list[str]


@dataclass
class NormalizationStats:
    surface_mean: np.ndarray
    surface_std: np.ndarray
    rate_mean: np.ndarray
    rate_std: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray


@dataclass
class DatasetSplits:
    train: "OptionDataset"
    val: "OptionDataset"
    test: "OptionDataset"
    normalizer: NormalizationStats


@dataclass
class DataLoaderSplits:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    normalizer: NormalizationStats


@dataclass
class WalkForwardFold:
    fold_id: int
    train: "OptionDataset"
    val: "OptionDataset"
    normalizer: NormalizationStats
    train_target_start: str
    train_target_end: str
    val_target_start: str
    val_target_end: str


@dataclass
class WalkForwardDatasetBundle:
    folds: list[WalkForwardFold]
    final_train: "OptionDataset"
    test: "OptionDataset"
    final_normalizer: NormalizationStats


@dataclass
class WalkForwardLoaderFold:
    fold_id: int
    train: DataLoader
    val: DataLoader
    normalizer: NormalizationStats
    train_target_start: str
    train_target_end: str
    val_target_start: str
    val_target_end: str


@dataclass
class WalkForwardLoaderBundle:
    folds: list[WalkForwardLoaderFold]
    final_train: DataLoader
    test: DataLoader
    final_normalizer: NormalizationStats


def load_daily_surface_data(data_path: Path) -> DailySurfaceData:
    df = pd.read_excel(data_path)

    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    required_cols = [
        "date",
        "sp500_close",
        "sp500_daily_simple_return",
        "sp500_open",
        "sp500_high",
        "sp500_low",
        "days_to_exp",
        "delta",
        "impl_volatility",
        "risk_free_rate",
        "vix_close",
        "vix_open",
        "vix_high",
        "vix_low",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "days_to_exp", "delta"]).reset_index(drop=True)

    null_counts = df[required_cols].isna().sum()
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        raise ValueError(f"Found nulls in cleaned dataframe:\n{null_counts}")

    duplicate_count = int(df.duplicated(subset=["date", "days_to_exp", "delta"]).sum())
    if duplicate_count > 0:
        raise ValueError(f"Found {duplicate_count} duplicate surface nodes.")

    n_dates = df["date"].nunique()

    # Keep only the surface grid that exists on every date.
    maturity_date_counts = df.groupby("days_to_exp")["date"].nunique()
    delta_date_counts = df.groupby("delta")["date"].nunique()

    stable_maturities = sorted(
        maturity_date_counts[maturity_date_counts == n_dates].index.tolist()
    )
    stable_deltas = sorted(
        delta_date_counts[delta_date_counts == n_dates].index.tolist()
    )

    if not stable_maturities or not stable_deltas:
        raise ValueError(
            "Could not find a stable surface grid present on every date.\n"
            f"Maturity coverage:\n{maturity_date_counts}\n"
            f"Delta coverage:\n{delta_date_counts}"
        )

    df = df[
        df["days_to_exp"].isin(stable_maturities)
        & df["delta"].isin(stable_deltas)
    ].copy()

    maturities = np.array(stable_maturities, dtype=np.float32)
    deltas = np.array(stable_deltas, dtype=np.float32)
    expected_nodes = len(maturities) * len(deltas)

    nodes_per_date = df.groupby("date").size()
    bad_dates = nodes_per_date[nodes_per_date != expected_nodes]
    if not bad_dates.empty:
        raise ValueError(
            "Some dates do not have the full surface grid.\n"
            f"Expected {expected_nodes} nodes per date.\n"
            f"Sample bad dates:\n{bad_dates.head(10)}"
        )

    feature_names = [
        "sp500_close",
        "sp500_daily_simple_return",
        "sp500_open",
        "sp500_high",
        "sp500_low",
        "vix_close",
        "vix_open",
        "vix_high",
        "vix_low",
    ]

    dates = []
    surfaces = []
    rate_curves = []
    daily_features = []

    for date, group in df.groupby("date", sort=True):
        group = group.sort_values(["days_to_exp", "delta"])

        surface = (
            group.pivot(index="days_to_exp", columns="delta", values="impl_volatility")
            .reindex(index=maturities, columns=deltas)
            .to_numpy(dtype=np.float32)
        )

        rate_curve = (
            group[["days_to_exp", "risk_free_rate"]]
            .drop_duplicates(subset=["days_to_exp"])
            .set_index("days_to_exp")
            .reindex(maturities)["risk_free_rate"]
            .to_numpy(dtype=np.float32)
        )

        feature_row = group[feature_names].iloc[0].to_numpy(dtype=np.float32)

        dates.append(date)
        surfaces.append(surface)
        rate_curves.append(rate_curve)
        daily_features.append(feature_row)

    return DailySurfaceData(
        dates=np.array(dates),
        maturities=maturities,
        deltas=deltas,
        surfaces=np.stack(surfaces),
        rate_curves=np.stack(rate_curves),
        daily_features=np.stack(daily_features),
        feature_names=feature_names,
    )


def _safe_std(array: np.ndarray) -> np.ndarray:
    return np.where(array < EPSILON, 1.0, array).astype(np.float32)


def _date_range_from_indices(
    dates: np.ndarray,
    sample_end_indices: np.ndarray,
) -> tuple[str, str]:
    target_indices = sample_end_indices + 1
    return (
        str(dates[int(target_indices[0])]),
        str(dates[int(target_indices[-1])]),
    )


def fit_normalization_stats(
    daily_data: DailySurfaceData,
    sample_end_indices: np.ndarray,
    lookback: int,
) -> NormalizationStats:
    sample_end_indices = np.asarray(sample_end_indices, dtype=int)
    if sample_end_indices.ndim != 1 or len(sample_end_indices) == 0:
        raise ValueError("sample_end_indices must be a non-empty 1D array")

    history_mask = np.zeros(len(daily_data.dates), dtype=bool)
    for history_end in sample_end_indices:
        history_start = history_end - lookback + 1
        history_mask[history_start : history_end + 1] = True

    surface_history = daily_data.surfaces[history_mask]
    rate_history = daily_data.rate_curves[history_mask]
    feature_history = daily_data.daily_features[history_mask]

    surface_delta = daily_data.surfaces[1:] - daily_data.surfaces[:-1]
    target_surface_delta = surface_delta[sample_end_indices]

    return NormalizationStats(
        surface_mean=surface_history.mean(axis=0).astype(np.float32),
        surface_std=_safe_std(surface_history.std(axis=0)),
        rate_mean=rate_history.mean(axis=0).astype(np.float32),
        rate_std=_safe_std(rate_history.std(axis=0)),
        feature_mean=feature_history.mean(axis=0).astype(np.float32),
        feature_std=_safe_std(feature_history.std(axis=0)),
        target_mean=target_surface_delta.mean(axis=0).astype(np.float32),
        target_std=_safe_std(target_surface_delta.std(axis=0)),
    )


def option_collate_fn(batch: list[dict[str, np.ndarray | str]]) -> dict[str, torch.Tensor | list[str]]:
    surface_history = torch.as_tensor(
        np.stack([item["surface_history"] for item in batch]),
        dtype=torch.float32,
    )
    rate_history = torch.as_tensor(
        np.stack([item["rate_history"] for item in batch]),
        dtype=torch.float32,
    )
    feature_history = torch.as_tensor(
        np.stack([item["feature_history"] for item in batch]),
        dtype=torch.float32,
    )
    target_surface_delta = torch.as_tensor(
        np.stack([item["target_surface_delta"] for item in batch]),
        dtype=torch.float32,
    )

    return {
        "surface_history": surface_history,
        "rate_history": rate_history,
        "feature_history": feature_history,
        "target_surface_delta": target_surface_delta,
        "history_end_date": [str(item["history_end_date"]) for item in batch],
        "target_date": [str(item["target_date"]) for item in batch],
    }


class OptionDataset(Dataset):
    def __init__(
        self,
        data_path: Path | str,
        lookback: int = 120,
        sample_end_indices: np.ndarray | None = None,
        daily_data: DailySurfaceData | None = None,
        normalization_stats: NormalizationStats | None = None,
    ):
        self.data_path = Path(data_path)
        self.lookback = lookback
        self.daily_data = (
            daily_data if daily_data is not None
            else load_daily_surface_data(self.data_path)
        )
        self.normalization_stats = normalization_stats
        self.surface_delta = (
            self.daily_data.surfaces[1:] - self.daily_data.surfaces[:-1]
        )
        num_dates = len(self.daily_data.dates)

        if self.lookback < 1:
            raise ValueError("lookback must be at least 1")
        if self.lookback >= num_dates:
            raise ValueError("lookback must be smaller than the number of dates")

        self.base_valid_end_indices = np.arange(self.lookback - 1, num_dates - 1)
        if sample_end_indices is None:
            self.sample_end_indices = self.base_valid_end_indices
        else:
            self.sample_end_indices = np.asarray(sample_end_indices, dtype=int)
            if self.sample_end_indices.ndim != 1:
                raise ValueError("sample_end_indices must be a 1D array")
            if len(self.sample_end_indices) == 0:
                raise ValueError("sample_end_indices cannot be empty")
            if np.any(self.sample_end_indices < self.lookback - 1):
                raise ValueError(
                    "sample_end_indices contain entries before the first valid history end"
                )
            if np.any(self.sample_end_indices > num_dates - 2):
                raise ValueError(
                    "sample_end_indices contain entries without a next-day target"
                )

        target_indices = self.sample_end_indices + 1
        self.active_date_indices = np.arange(
            int(target_indices.min()),
            int(target_indices.max()) + 1,
        )

    def __len__(self) -> int:
        return len(self.sample_end_indices)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | str]:
        history_end = int(self.sample_end_indices[idx])
        history_start = history_end - self.lookback + 1
        target_idx = history_end + 1

        surface_history = self.daily_data.surfaces[history_start : history_end + 1].copy()
        rate_history = self.daily_data.rate_curves[history_start : history_end + 1].copy()
        feature_history = self.daily_data.daily_features[history_start : history_end + 1].copy()
        target_surface_delta = self.surface_delta[history_end].copy()

        if self.normalization_stats is not None:
            surface_history = (
                (surface_history - self.normalization_stats.surface_mean[None, :, :])
                / self.normalization_stats.surface_std[None, :, :]
            ).astype(np.float32)
            rate_history = (
                (rate_history - self.normalization_stats.rate_mean[None, :])
                / self.normalization_stats.rate_std[None, :]
            ).astype(np.float32)
            feature_history = (
                (feature_history - self.normalization_stats.feature_mean[None, :])
                / self.normalization_stats.feature_std[None, :]
            ).astype(np.float32)
            target_surface_delta = (
                (target_surface_delta - self.normalization_stats.target_mean)
                / self.normalization_stats.target_std
            ).astype(np.float32)

        return {
            "surface_history": surface_history,
            "rate_history": rate_history,
            "feature_history": feature_history,
            "target_surface_delta": target_surface_delta,
            "history_end_date": str(self.daily_data.dates[history_end]),
            "target_date": str(self.daily_data.dates[target_idx]),
        }

    @classmethod
    def create_splits(
        cls,
        data_path: Path | str,
        lookback: int = 120,
        split: tuple[float, float, float] = (0.70, 0.15, 0.15),
    ) -> DatasetSplits:
        daily_data = load_daily_surface_data(Path(data_path))
        train_indices, val_indices, test_indices = make_split_indices(
            num_dates=len(daily_data.dates),
            lookback=lookback,
            split=split,
        )
        normalizer = fit_normalization_stats(daily_data, train_indices, lookback)

        return DatasetSplits(
            train=cls(
                data_path,
                lookback=lookback,
                sample_end_indices=train_indices,
                daily_data=daily_data,
                normalization_stats=normalizer,
            ),
            val=cls(
                data_path,
                lookback=lookback,
                sample_end_indices=val_indices,
                daily_data=daily_data,
                normalization_stats=normalizer,
            ),
            test=cls(
                data_path,
                lookback=lookback,
                sample_end_indices=test_indices,
                daily_data=daily_data,
                normalization_stats=normalizer,
            ),
            normalizer=normalizer,
        )

    @classmethod
    def create_walk_forward_splits(
        cls,
        data_path: Path | str,
        lookback: int = 120,
        test_fraction: float = 0.15,
        val_size: int = 252,
        step_size: int = 252,
        min_train_size: int | None = None,
    ) -> WalkForwardDatasetBundle:
        daily_data = load_daily_surface_data(Path(data_path))
        fold_indices, final_train_indices, test_indices = make_walk_forward_indices(
            num_dates=len(daily_data.dates),
            lookback=lookback,
            test_fraction=test_fraction,
            val_size=val_size,
            step_size=step_size,
            min_train_size=min_train_size,
        )

        folds = []
        for fold_id, (train_indices, val_indices) in enumerate(fold_indices):
            normalizer = fit_normalization_stats(daily_data, train_indices, lookback)
            train_target_start, train_target_end = _date_range_from_indices(
                daily_data.dates,
                train_indices,
            )
            val_target_start, val_target_end = _date_range_from_indices(
                daily_data.dates,
                val_indices,
            )

            folds.append(
                WalkForwardFold(
                    fold_id=fold_id,
                    train=cls(
                        data_path,
                        lookback=lookback,
                        sample_end_indices=train_indices,
                        daily_data=daily_data,
                        normalization_stats=normalizer,
                    ),
                    val=cls(
                        data_path,
                        lookback=lookback,
                        sample_end_indices=val_indices,
                        daily_data=daily_data,
                        normalization_stats=normalizer,
                    ),
                    normalizer=normalizer,
                    train_target_start=train_target_start,
                    train_target_end=train_target_end,
                    val_target_start=val_target_start,
                    val_target_end=val_target_end,
                )
            )

        final_normalizer = fit_normalization_stats(
            daily_data,
            final_train_indices,
            lookback,
        )

        return WalkForwardDatasetBundle(
            folds=folds,
            final_train=cls(
                data_path,
                lookback=lookback,
                sample_end_indices=final_train_indices,
                daily_data=daily_data,
                normalization_stats=final_normalizer,
            ),
            test=cls(
                data_path,
                lookback=lookback,
                sample_end_indices=test_indices,
                daily_data=daily_data,
                normalization_stats=final_normalizer,
            ),
            final_normalizer=final_normalizer,
        )


def make_split_indices(
    num_dates: int,
    lookback: int = 120,
    split: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_frac, val_frac, test_frac = split
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("split fractions must sum to 1.0")
    if lookback < 1:
        raise ValueError("lookback must be at least 1")
    if num_dates <= lookback:
        raise ValueError("num_dates must be greater than lookback")

    base_valid_end_indices = np.arange(lookback - 1, num_dates - 1)

    train_end = int(num_dates * train_frac)
    val_end = train_end + int(num_dates * val_frac)

    train_indices = base_valid_end_indices[base_valid_end_indices + 1 < train_end]
    val_indices = base_valid_end_indices[
        (base_valid_end_indices + 1 >= train_end)
        & (base_valid_end_indices + 1 < val_end)
    ]
    test_indices = base_valid_end_indices[base_valid_end_indices + 1 >= val_end]

    if len(train_indices) == 0 or len(val_indices) == 0 or len(test_indices) == 0:
        raise ValueError(
            "Split produced an empty dataset. Adjust lookback or split fractions."
        )

    return train_indices, val_indices, test_indices


def make_walk_forward_indices(
    num_dates: int,
    lookback: int = 120,
    test_fraction: float = 0.15,
    val_size: int = 252,
    step_size: int = 252,
    min_train_size: int | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    if lookback < 1:
        raise ValueError("lookback must be at least 1")
    if num_dates <= lookback:
        raise ValueError("num_dates must be greater than lookback")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if val_size < 1 or step_size < 1:
        raise ValueError("val_size and step_size must be positive")

    if min_train_size is None:
        min_train_size = max(lookback, val_size * 3)
    if min_train_size < lookback:
        raise ValueError("min_train_size must be at least lookback")

    base_valid_end_indices = np.arange(lookback - 1, num_dates - 1)
    test_target_start = int(num_dates * (1.0 - test_fraction))

    pretest_indices = base_valid_end_indices[base_valid_end_indices + 1 < test_target_start]
    test_indices = base_valid_end_indices[base_valid_end_indices + 1 >= test_target_start]

    if len(pretest_indices) == 0 or len(test_indices) == 0:
        raise ValueError("Walk-forward split produced an empty pre-test or test set.")

    first_val_target_start = min_train_size
    last_val_target_start = test_target_start - val_size
    if first_val_target_start > last_val_target_start:
        raise ValueError(
            "Not enough pre-test data to create a walk-forward validation fold. "
            "Reduce val_size/test_fraction or min_train_size."
        )

    fold_starts = list(range(first_val_target_start, last_val_target_start + 1, step_size))
    if fold_starts[-1] != last_val_target_start:
        fold_starts.append(last_val_target_start)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for val_target_start in fold_starts:
        val_target_end = val_target_start + val_size
        train_indices = pretest_indices[pretest_indices + 1 < val_target_start]
        val_indices = pretest_indices[
            (pretest_indices + 1 >= val_target_start)
            & (pretest_indices + 1 < val_target_end)
        ]
        if len(train_indices) == 0 or len(val_indices) == 0:
            continue
        folds.append((train_indices, val_indices))

    if not folds:
        raise ValueError("No valid walk-forward folds were created.")

    final_train_indices = pretest_indices
    return folds, final_train_indices, test_indices


def build_dataloaders(
    data_path: Path | str,
    lookback: int = 120,
    batch_size: int = 32,
    split: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> DataLoaderSplits:
    datasets = OptionDataset.create_splits(
        data_path=data_path,
        lookback=lookback,
        split=split,
    )

    train_loader = DataLoader(
        datasets.train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=option_collate_fn,
    )
    val_loader = DataLoader(
        datasets.val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=option_collate_fn,
    )
    test_loader = DataLoader(
        datasets.test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=option_collate_fn,
    )

    return DataLoaderSplits(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        normalizer=datasets.normalizer,
    )


def build_walk_forward_dataloaders(
    data_path: Path | str,
    lookback: int = 120,
    batch_size: int = 32,
    test_fraction: float = 0.15,
    val_size: int = 252,
    step_size: int = 252,
    min_train_size: int | None = None,
) -> WalkForwardLoaderBundle:
    datasets = OptionDataset.create_walk_forward_splits(
        data_path=data_path,
        lookback=lookback,
        test_fraction=test_fraction,
        val_size=val_size,
        step_size=step_size,
        min_train_size=min_train_size,
    )

    fold_loaders = []
    for fold in datasets.folds:
        fold_loaders.append(
            WalkForwardLoaderFold(
                fold_id=fold.fold_id,
                train=DataLoader(
                    fold.train,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=option_collate_fn,
                ),
                val=DataLoader(
                    fold.val,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=option_collate_fn,
                ),
                normalizer=fold.normalizer,
                train_target_start=fold.train_target_start,
                train_target_end=fold.train_target_end,
                val_target_start=fold.val_target_start,
                val_target_end=fold.val_target_end,
            )
        )

    final_train_loader = DataLoader(
        datasets.final_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=option_collate_fn,
    )
    test_loader = DataLoader(
        datasets.test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=option_collate_fn,
    )

    return WalkForwardLoaderBundle(
        folds=fold_loaders,
        final_train=final_train_loader,
        test=test_loader,
        final_normalizer=datasets.final_normalizer,
    )


if __name__ == "__main__":
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = PROJECT_DIR / "data" / "raw" / "all_data.xlsx"

    datasets = OptionDataset.create_walk_forward_splits(
        DATA_PATH,
        lookback=252,
        test_fraction=0.15,
        val_size=90,
        step_size=90,
    )
    first_fold = datasets.folds[0]
    sample = first_fold.train[0]

    print(f"Walk-forward folds: {len(datasets.folds)}")
    print(f"First fold train length: {len(first_fold.train)}")
    print(f"First fold val length: {len(first_fold.val)}")
    print(f"Final train length: {len(datasets.final_train)}")
    print(f"Test length: {len(datasets.test)}")
    print(f"Sample keys: {list(sample.keys())}")
    print(f"surface_history shape: {sample['surface_history'].shape}")
    print(f"rate_history shape: {sample['rate_history'].shape}")
    print(f"feature_history shape: {sample['feature_history'].shape}")
    print(f"target_surface_delta shape: {sample['target_surface_delta'].shape}")
    print(
        "First fold target ranges:",
        first_fold.train_target_start,
        first_fold.train_target_end,
        first_fold.val_target_start,
        first_fold.val_target_end,a
    )
