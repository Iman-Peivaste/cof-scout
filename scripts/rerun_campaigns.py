#!/usr/bin/env python3
"""Explicit live-run entry point; requires endpoint credentials and may incur costs."""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--replicate", type=int, required=True)
    args = parser.parse_args()
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY"):
        if not os.getenv(name):
            parser.error(f"{name} is required")
    from cof_scout.agent.loop import public_main

    public_main(config_path=args.config, replicate=args.replicate)


if __name__ == "__main__":
    main()
