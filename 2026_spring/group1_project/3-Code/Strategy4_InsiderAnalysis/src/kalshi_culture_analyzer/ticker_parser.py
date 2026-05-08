from __future__ import annotations

from dataclasses import dataclass
from typing import List
import re


@dataclass(frozen=True)
class TickerEntry:
    ticker: str
    tags: List[str]
    raw_line: str


_TAG_SPLIT_RE = re.compile(r"[\s,;]+")


def parse_ticker_file(path: str) -> List[TickerEntry]:
    entries: List[TickerEntry] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw_line = raw.rstrip("\n")
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith("//"):
                continue
            # Strip inline // comments
            if "//" in line:
                line = line.split("//", 1)[0].strip()
                if not line:
                    continue
            tags: List[str] = []
            if "#" in line:
                ticker_part, tag_part = line.split("#", 1)
                ticker = ticker_part.strip()
                tags = [t for t in _TAG_SPLIT_RE.split(tag_part.strip()) if t]
            else:
                ticker = line

            if not ticker:
                continue
            entries.append(TickerEntry(ticker=ticker, tags=tags, raw_line=raw_line))

    return entries

