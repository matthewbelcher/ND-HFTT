from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

NUM_RE = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"


@dataclass(frozen=True)
class RangeParse:
    raw_low: Optional[float]
    raw_high: Optional[float]
    norm_low: Optional[int]
    norm_high: Optional[int]
    text: str
    decimal_warning: Optional[str]


@dataclass(frozen=True)
class ThresholdParse:
    raw_strike: Optional[float]
    norm_strike: Optional[int]
    text: str
    decimal_warning: Optional[str]


@dataclass(frozen=True)
class Classification:
    kind: str  # "bucket", "threshold", "ignore"
    range_parse: Optional[RangeParse] = None
    threshold_parse: Optional[ThresholdParse] = None
    source_field: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


def _parse_num(s: str) -> float:
    return float(s.replace(",", ""))


def _norm_int(v: Optional[float]) -> Optional[int]:
    if v is None:
        return None
    if abs(v - round(v)) < 1e-9:
        return int(round(v))
    return None


def _frac_part(v: float) -> float:
    return abs(v - int(v))


def _is_intish(v: Optional[float]) -> bool:
    if v is None:
        return True
    return _frac_part(v) < 1e-6


def _is_upper_tolerable(v: Optional[float]) -> bool:
    if v is None:
        return True
    frac = _frac_part(v)
    return frac < 1e-6 or abs(frac - 0.99) < 1e-6


def _range_from_match(text: str, a_str: str, b_str: str) -> RangeParse:
    a = _parse_num(a_str)
    b = _parse_num(b_str)
    warn = None
    # Warn if lower bound has decimals, or upper bound has unusual decimals.
    if not _is_intish(a) or not _is_upper_tolerable(b):
        warn = f"Decimal value detected in text; normalization may discard decimals: '{text}'."
    return RangeParse(
        raw_low=a,
        raw_high=b,
        norm_low=_norm_int(a),
        norm_high=_norm_int(b),
        text=text,
        decimal_warning=warn,
    )


def _open_range_from_match(
    text: str,
    low_str: Optional[str],
    high_str: Optional[str],
) -> RangeParse:
    raw_low = _parse_num(low_str) if low_str is not None else None
    raw_high = _parse_num(high_str) if high_str is not None else None
    warn = None
    if raw_low is not None and not _is_intish(raw_low):
        warn = f"Decimal value detected in text; normalization may discard decimals: '{text}'."
    if raw_high is not None and not _is_upper_tolerable(raw_high):
        warn = f"Decimal value detected in text; normalization may discard decimals: '{text}'."
    return RangeParse(
        raw_low=raw_low,
        raw_high=raw_high,
        norm_low=_norm_int(raw_low),
        norm_high=_norm_int(raw_high),
        text=text,
        decimal_warning=warn,
    )


def parse_range_text(text: Optional[str]) -> Optional[RangeParse]:
    if not text:
        return None
    s = text.strip().replace("$", "")
    if not s:
        return None

    range_patterns = [
        rf"between\s+({NUM_RE})\s+and\s+({NUM_RE})",
        rf"({NUM_RE})\s*(?:to|–|—|-)\s*({NUM_RE})",
    ]
    for pat in range_patterns:
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            groups = [g for g in m.groups() if g is not None]
            if len(groups) >= 2:
                return _range_from_match(s, groups[0], groups[1])

    low_patterns = [
        rf"(?:at\s+or\s+above|at\s+least|>=|above|over)\s*({NUM_RE})",
        rf"({NUM_RE})\s*(?:\+|or\s+higher|or\s+above)",
    ]
    for pat in low_patterns:
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            groups = [g for g in m.groups() if g is not None]
            if groups:
                return _open_range_from_match(s, groups[0], None)

    high_patterns = [
        rf"(?:at\s+or\s+below|<=|under|below|less\s+than)\s*({NUM_RE})",
        rf"({NUM_RE})\s*(?:or\s+below|or\s+less)",
    ]
    for pat in high_patterns:
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            groups = [g for g in m.groups() if g is not None]
            if groups:
                return _open_range_from_match(s, None, groups[0])

    return None


def parse_threshold_text(text: Optional[str]) -> Optional[ThresholdParse]:
    if not text:
        return None
    s = text.strip().replace("$", "")
    if not s:
        return None

    # Reject <= / below style
    if re.search(r"(?:<=|below|under|at\s+or\s+below|or\s+less)", s, re.IGNORECASE):
        return None

    ge_patterns = [
        rf"(?:>=|at\s+or\s+above|at\s+least)\s*({NUM_RE})",
        rf"({NUM_RE})\s*(?:or\s+higher|or\s+above|\+)",
    ]
    for pat in ge_patterns:
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            groups = [g for g in m.groups() if g is not None]
            if not groups:
                continue
            num_str = groups[0]
            raw = _parse_num(num_str)
            warn = None
            if not _is_intish(raw):
                warn = f"Decimal value detected in text; normalization may discard decimals: '{s}'."
            return ThresholdParse(
                raw_strike=raw,
                norm_strike=_norm_int(raw),
                text=s,
                decimal_warning=warn,
            )

    return None


def classify_market(market: dict, prefer: Optional[str] = None) -> Classification:
    fields: List[Tuple[str, Optional[str]]] = [
        ("subtitle", market.get("subtitle")),
        ("yes_sub_title", market.get("yes_sub_title")),
        ("title", market.get("title")),
        ("rules_primary", market.get("rules_primary")),
        ("ticker", market.get("ticker")),
    ]
    for field_name, text in fields:
        if not text:
            continue
        rng = parse_range_text(text)
        thr = parse_threshold_text(text)
        warnings: List[str] = []
        if rng and rng.decimal_warning:
            warnings.append(rng.decimal_warning)
        if thr and thr.decimal_warning:
            warnings.append(thr.decimal_warning)

        if prefer == "threshold":
            if thr:
                return Classification(
                    kind="threshold",
                    threshold_parse=thr,
                    source_field=field_name,
                    warnings=warnings,
                )
            if rng:
                return Classification(
                    kind="bucket",
                    range_parse=rng,
                    source_field=field_name,
                    warnings=warnings,
                )
        elif prefer == "bucket":
            if rng:
                return Classification(
                    kind="bucket",
                    range_parse=rng,
                    source_field=field_name,
                    warnings=warnings,
                )
            if thr:
                return Classification(
                    kind="threshold",
                    threshold_parse=thr,
                    source_field=field_name,
                    warnings=warnings,
                )
        else:
            if rng:
                return Classification(
                    kind="bucket",
                    range_parse=rng,
                    source_field=field_name,
                    warnings=warnings,
                )
            if thr:
                return Classification(
                    kind="threshold",
                    threshold_parse=thr,
                    source_field=field_name,
                    warnings=warnings,
                )

    return Classification(kind="ignore")
