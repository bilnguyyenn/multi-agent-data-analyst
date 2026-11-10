"""Verify and regenerate published historical scorecards without network/model access."""

from __future__ import annotations

import json
import re
from pathlib import Path

from analyst.eval.harness import run_integrity_errors
from analyst.eval.heldout_harness import heldout_integrity_errors, render_heldout_scorecard
from analyst.eval.scorecard import render


def reproduce(root: Path, output: Path) -> dict:
    main_path = root / "eval_results/SCORECARD.md"
    held_path = root / "heldout_eval_results/SCORECARD.md"
    main_text, held_text = main_path.read_text(), held_path.read_text()
    sources = dict(re.findall(r"^- `([^`]+)`: `(eval_results/raw_[^`]+\.json)`", main_text, re.M))
    held_sources = {
        (name, int(seed)): path
        for name, seed, path in re.findall(
            r"^- `([^`]+)`, seed `(\d+)`: `(heldout_eval_results/heldout_raw_[^`]+\.json)`",
            held_text,
            re.M,
        )
    }
    if len(sources) != 7 or len(held_sources) != 2:
        raise ValueError("historical scorecard provenance contract changed")
    runs, held_runs = [], []
    for mapping, dest, gate in [
        (sources, runs, run_integrity_errors),
        (held_sources, held_runs, heldout_integrity_errors),
    ]:
        for key, relative in mapping.items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("provenance path escapes repository")
            saved = json.loads(path.read_text())
            matches = [
                r
                for r in saved
                if r.get("config") == (key[0] if isinstance(key, tuple) else key)
                and (not isinstance(key, tuple) or r.get("seed") == key[1])
            ]
            if len(matches) != 1 or gate(matches[0]):
                raise ValueError(f"invalid cited run: {relative}")
            # Reviewed synthetic evaluation outputs must never publish credentials.
            if re.search(
                r"(?:sk-[A-Za-z0-9_-]{20,}|-----BEGIN .*PRIVATE KEY-----)", path.read_text()
            ):
                raise ValueError(f"credential-shaped content in {relative}")
            dest.append(matches[0])
    critic = json.loads((root / "eval_results/critic_bench.json").read_text())
    regenerated = render(runs, critic, sources=sources)
    regenerated_held = render_heldout_scorecard(held_runs, sources=held_sources)

    def rows(text):
        return sorted(line for line in text.splitlines() if line.startswith("|"))

    if rows(regenerated) != rows(main_text) or rows(regenerated_held) != rows(held_text):
        raise ValueError("published metric tables differ from their raw inputs")
    output.mkdir(parents=True, exist_ok=True)
    (output / "DEVELOPMENT.md").write_text(regenerated)
    (output / "HELDOUT.md").write_text(regenerated_held)
    result = {
        "development_runs": len(runs),
        "heldout_runs": len(held_runs),
        "provenance_files": list(sources.values()) + list(held_sources.values()),
        "status": "verified",
        "scope": "historical saved metrics, not a new live evaluation",
    }
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
