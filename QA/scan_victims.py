#!/usr/bin/env python
"""Scan MMLU subject CSVs for (subject, victim) combos with enough hits.

Run from QA/ directory:
    python scan_victims.py
    python scan_victims.py --min-hits 30 --victims food security drug

For each candidate victim term we count how many test questions in each MMLU
subject contain it (word-boundary, case-insensitive). Combos with at least
`--min-hits` matches are printed — those are the ones safe to use with
`num_templates + num_test` up to that count.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from glob import glob

DEFAULT_VICTIMS = [
    "food", "security", "law", "virus", "tax",
    "energy", "system", "study", "test", "data",
    "value", "drug", "growth", "government", "market",
    "cell", "force", "function", "rate", "income",
]


def count_hits(question: str, victim: str) -> int:
    pattern = re.compile(r"\b" + re.escape(victim) + r"s?\b", re.IGNORECASE)
    return 1 if pattern.search(question) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/test")
    ap.add_argument("--min-hits", type=int, default=20,
                    help="Only show combos with at least this many matching questions "
                         "(default 20 = 10 templates + 10 test, paper MMLU per-pair).")
    ap.add_argument("--victims", nargs="*", default=DEFAULT_VICTIMS,
                    help="Candidate victim terms to test.")
    args = ap.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"data dir not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    by_victim: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for csv_path in sorted(glob(os.path.join(args.data_dir, "*_test.csv"))):
        subject = os.path.basename(csv_path).removesuffix("_test.csv")
        with open(csv_path, newline="") as f:
            rows = [r for r in csv.reader(f) if r]
        questions = [r[0] for r in rows]
        for v in args.victims:
            hits = sum(count_hits(q, v) for q in questions)
            if hits >= args.min_hits:
                by_victim[v].append((subject, hits))

    if not by_victim:
        print(f"No (subject, victim) combos with >= {args.min_hits} hits.")
        return

    print(f"Viable (subject, victim) combos (>= {args.min_hits} hits):\n")
    for v in sorted(by_victim):
        hits_sorted = sorted(by_victim[v], key=lambda x: -x[1])
        print(f"  victim = {v!r}")
        for subj, n in hits_sorted[:5]:
            print(f"    {n:>4d}  {subj}_test.csv")
        print()


if __name__ == "__main__":
    main()
