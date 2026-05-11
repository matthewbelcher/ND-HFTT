from __future__ import annotations

import argparse
import sys

from .collector import collect
from .config import Config
from .export import export_db
from .log import configure_logging
from .report import generate_report
from .replay import replay_scores
from .resolve import resolve_outcomes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kalshi-analyzer", description="Kalshi Culture/Event Flow Analyzer")
    parser.add_argument("--config", help="Path to config.yaml")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect", help="Collect live data and build dataset")
    p_collect.add_argument("--tickers", required=True, help="Path to tickers.txt")
    p_collect.add_argument("--poll-minutes", type=float, help="Polling interval in minutes")
    p_collect.add_argument("--db", required=True, help="SQLite DB path")
    p_collect.add_argument("--historical", action="store_true", help="Use historical endpoints when possible")

    p_report = sub.add_parser("report", help="Generate summary reports")
    p_report.add_argument("--db", required=True, help="SQLite DB path")
    p_report.add_argument("--event", help="Event ticker to filter")
    p_report.add_argument("--top", type=int, default=10, help="Top K markets to show")
    p_report.add_argument("--since", help="Relative duration (e.g., 7d) or ISO date")

    p_resolve = sub.add_parser("resolve", help="Update outcomes for closed markets")
    p_resolve.add_argument("--db", required=True, help="SQLite DB path")

    p_replay = sub.add_parser("replay", help="Recompute features/scores from stored raw data")
    p_replay.add_argument("--db", required=True, help="SQLite DB path")
    p_replay.add_argument("--rules-config", help="Path to config.yaml to override scoring rules")

    p_export = sub.add_parser("export", help="Export dataset to CSV or Parquet")
    p_export.add_argument("--db", required=True, help="SQLite DB path")
    p_export.add_argument("--format", choices=["csv", "parquet"], required=True)
    p_export.add_argument("--out", required=True, help="Output directory")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    config_path = getattr(args, "config", None)
    if args.command == "replay" and args.rules_config:
        config_path = args.rules_config

    config = Config.from_yaml(config_path)

    if args.command == "collect":
        collect(
            tickers_file=args.tickers,
            db_path=args.db,
            config=config,
            poll_minutes=args.poll_minutes,
            use_historical=args.historical,
        )
    elif args.command == "report":
        generate_report(db_path=args.db, event_ticker=args.event, top_k=args.top, since=args.since, config=config)
    elif args.command == "resolve":
        resolve_outcomes(db_path=args.db, config=config)
    elif args.command == "replay":
        replay_scores(db_path=args.db, config=config)
    elif args.command == "export":
        export_db(db_path=args.db, out_dir=args.out, fmt=args.format)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

