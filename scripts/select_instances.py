#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    p = argparse.ArgumentParser(description="Select a reproducible subset of SWE-bench instances.")
    p.add_argument("--dataset_name", default="princeton-nlp/SWE-bench_Lite")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out", default="outputs/instance_ids.txt")
    p.add_argument("--manifest", default="outputs/selection_manifest.json")
    args = p.parse_args()

    ds = load_dataset(args.dataset_name, split=args.split)
    ids = [row["instance_id"] for row in ds]
    random.Random(args.seed).shuffle(ids)
    selected = sorted(ids[: args.limit])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(selected) + "\n", encoding="utf-8")

    manifest = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "limit": args.limit,
        "seed": args.seed,
        "selected_count": len(selected),
        "instance_ids_file": str(out),
        "instance_ids": selected,
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(selected)} instance ids to {out}")


if __name__ == "__main__":
    main()
