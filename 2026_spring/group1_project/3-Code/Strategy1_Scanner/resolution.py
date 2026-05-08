from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResolutionSpec:
    """How many markets in an event are expected to resolve YES."""

    yes_count: int | None
    source: str
    reason: str

    @property
    def is_exact(self) -> bool:
        return self.yes_count is not None


UNKNOWN_RESOLUTION = ResolutionSpec(
    yes_count=None,
    source="unknown",
    reason="No exact winner count could be determined.",
)

DEFAULT_RESOLUTION_CONFIG = "resolution_overrides.json"


def load_resolution_overrides(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _normalize_ticker(value: Any) -> str:
    return str(value or "").strip().upper()


def _override_lookup(config: dict[str, Any], event: dict[str, Any]) -> ResolutionSpec | None:
    event_ticker = _normalize_ticker(event.get("event_ticker") or event.get("ticker"))
    series_ticker = _normalize_ticker(event.get("series_ticker"))

    candidates: list[tuple[str, Any]] = []
    if event_ticker:
        candidates.extend(
            [
                (f"events.{event_ticker}", (config.get("events") or {}).get(event_ticker)),
                (event_ticker, config.get(event_ticker)),
            ]
        )
    if series_ticker:
        candidates.extend(
            [
                (f"series.{series_ticker}", (config.get("series") or {}).get(series_ticker)),
                (series_ticker, config.get(series_ticker)),
            ]
        )

    for source, entry in candidates:
        if entry is None:
            continue
        if isinstance(entry, int):
            return ResolutionSpec(entry, "config", f"{source} sets yes_count={entry}.")
        if isinstance(entry, dict):
            raw = entry.get("yes_count", entry.get("n_yes", entry.get("winners")))
            try:
                count = int(raw)
            except (TypeError, ValueError):
                continue
            return ResolutionSpec(count, "config", f"{source} sets yes_count={count}.")
    return None


def _iter_text_fields(event: dict[str, Any], markets: list[dict[str, Any]]) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    for key in (
        "event_ticker",
        "ticker",
        "series_ticker",
        "title",
        "sub_title",
        "category",
        "strike_period",
    ):
        val = event.get(key)
        if val:
            fields.append((f"event.{key}", str(val)))

    metadata = event.get("product_metadata")
    if isinstance(metadata, dict):
        for key, val in metadata.items():
            if val:
                fields.append((f"event.product_metadata.{key}", str(val)))

    for i, market in enumerate(markets[:8]):
        for key in ("ticker", "title", "subtitle", "rules_primary", "rules_secondary"):
            val = market.get(key)
            if val:
                fields.append((f"market[{i}].{key}", str(val)))
    return fields


def _extract_top_n_counts(fields: list[tuple[str, str]]) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    patterns = [
        re.compile(r"TOP\s*[-_]?\s*(\d{1,2})\b", re.IGNORECASE),
        re.compile(r"\bTOP\s+(\d{1,2})\b", re.IGNORECASE),
        re.compile(r"\bFINISH(?:ES)?\s+IN\s+THE\s+TOP\s+(\d{1,2})\b", re.IGNORECASE),
        re.compile(r"\bTOP\s+(\d{1,2})\s+FINISHERS?\b", re.IGNORECASE),
    ]
    for source, text in fields:
        for pattern in patterns:
            for match in pattern.finditer(text):
                out.append((int(match.group(1)), source))
    return out


def _infer_top_n(fields: list[tuple[str, str]], total_markets: int) -> ResolutionSpec | None:
    hits = _extract_top_n_counts(fields)
    if not hits:
        return None
    counts = {count for count, _ in hits}
    if len(counts) != 1:
        detail = ", ".join(f"{count} from {source}" for count, source in hits)
        return ResolutionSpec(None, "conflict", f"Conflicting TOP-N counts: {detail}.")
    count = counts.pop()
    if count <= 0 or count > total_markets:
        return ResolutionSpec(None, "invalid", f"TOP-{count} is outside total market count {total_markets}.")
    sources = sorted({source for _, source in hits})
    return ResolutionSpec(count, "top_n_inferred", f"Found TOP-{count} in {', '.join(sources)}.")


def _infer_medal_event(fields: list[tuple[str, str]], total_markets: int) -> ResolutionSpec | None:
    joined = "\n".join(text for _, text in fields)
    has_any_medal = re.search(r"\bANY\s+MEDAL\b|\bWINS?\s+ANY\s+MEDAL\b", joined, re.IGNORECASE)
    has_medal_winner = re.search(r"\bMEDAL\s+WINNERS?\b", joined, re.IGNORECASE)
    has_specific_medal = re.search(r"\b(?:GOLD|SILVER|BRONZE)\s+MEDAL\b", joined, re.IGNORECASE)
    if has_specific_medal and not has_any_medal:
        return None
    if not (has_any_medal or has_medal_winner):
        return None
    if total_markets < 3:
        return ResolutionSpec(None, "invalid", f"Medal event has only {total_markets} markets.")
    return ResolutionSpec(3, "medal_inferred", "Medal-market wording implies gold/silver/bronze winners.")


def infer_resolution_spec(
    event: dict[str, Any],
    markets: list[dict[str, Any]],
    *,
    override_yes_count: int | None = None,
    config: dict[str, Any] | None = None,
    allow_inferred: bool = True,
) -> ResolutionSpec:
    total_markets = len(markets)
    if override_yes_count is not None:
        if 0 <= override_yes_count <= total_markets:
            return ResolutionSpec(override_yes_count, "cli", f"CLI override sets yes_count={override_yes_count}.")
        return ResolutionSpec(
            None,
            "invalid",
            f"CLI yes_count={override_yes_count} is outside total market count {total_markets}.",
        )

    if config:
        spec = _override_lookup(config, event)
        if spec is not None:
            if spec.yes_count is not None and 0 <= spec.yes_count <= total_markets:
                return spec
            return ResolutionSpec(
                None,
                "invalid",
                f"Configured yes_count={spec.yes_count} is outside total market count {total_markets}.",
            )

    if event.get("mutually_exclusive", False):
        return ResolutionSpec(1, "kalshi_mutually_exclusive", "Kalshi marks the event mutually_exclusive.")

    if not allow_inferred:
        return UNKNOWN_RESOLUTION

    fields = _iter_text_fields(event, markets)
    for inference in (_infer_top_n(fields, total_markets), _infer_medal_event(fields, total_markets)):
        if inference is None:
            continue
        return inference

    return UNKNOWN_RESOLUTION
