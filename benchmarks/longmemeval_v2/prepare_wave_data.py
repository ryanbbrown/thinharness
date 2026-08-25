#!/usr/bin/env python3
"""Prepare domain-small trajectory files without changing the public dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .freeze_selection import sha256_file


def relative_symlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if target.resolve() != source.resolve():
            raise RuntimeError(f"Existing path points elsewhere: {target}")
        return
    target.symlink_to(os.path.relpath(source, start=target.parent), target_is_directory=source.is_dir())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(original_data_root: Path, selection_path: Path, output_root: Path) -> dict[str, Any]:
    selection = load_json(selection_path)
    haystacks = load_json(original_data_root / "haystacks" / "lme_v2_small.json")
    ids_by_domain: dict[str, list[str]] = {}
    for domain in ("web", "enterprise"):
        domain_questions = [
            row["question_id"] for row in selection["questions"] if row["domain"] == domain
        ]
        first_ids = list(haystacks[domain_questions[0]])
        if len(first_ids) != 100:
            raise RuntimeError(f"Expected 100 Small trajectories for {domain}")
        if any(list(haystacks[question_id]) != first_ids for question_id in domain_questions):
            raise RuntimeError(f"Selected {domain} questions do not share the Small haystack")
        ids_by_domain[domain] = first_ids

    wanted = {item for values in ids_by_domain.values() for item in values}
    trajectories: dict[str, str] = {}
    with (original_data_root / "trajectories.jsonl").open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id") in wanted:
                trajectories[row["id"]] = line if line.endswith("\n") else line + "\n"
    missing = wanted - trajectories.keys()
    if missing:
        raise RuntimeError(f"Missing selected trajectories: {sorted(missing)[:5]}")

    domains: dict[str, Any] = {}
    for domain, trajectory_ids in ids_by_domain.items():
        domain_root = output_root / domain
        domain_root.mkdir(parents=True, exist_ok=True)
        relative_symlink(original_data_root / "questions.jsonl", domain_root / "questions.jsonl")
        relative_symlink(original_data_root / "haystacks", domain_root / "haystacks")
        relative_symlink(original_data_root / "screenshots", domain_root / "screenshots")
        relative_symlink(
            original_data_root / "question_screenshots",
            domain_root / "question_screenshots",
        )
        trajectories_path = domain_root / "trajectories.jsonl"
        expected_text = "".join(trajectories[item] for item in trajectory_ids)
        if trajectories_path.exists():
            if trajectories_path.read_text(encoding="utf-8") != expected_text:
                raise RuntimeError(f"Prepared trajectory file differs: {trajectories_path}")
        else:
            trajectories_path.write_text(expected_text, encoding="utf-8")
        domains[domain] = {
            "data_root": str(domain_root.resolve()),
            "trajectory_count": len(trajectory_ids),
            "trajectory_ids_sha256": __import__("hashlib").sha256(
                "\n".join(trajectory_ids).encode()
            ).hexdigest(),
            "trajectories_jsonl_sha256": sha256_file(trajectories_path),
            "trajectories_jsonl_bytes": trajectories_path.stat().st_size,
        }
    manifest = {
        "schema_version": 1,
        "source_data_root": str(original_data_root.resolve()),
        "source_questions_sha256": sha256_file(original_data_root / "questions.jsonl"),
        "selection_sha256": sha256_file(selection_path),
        "domains": domains,
    }
    manifest_path = output_root / "prepared_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(
        args.source_data_root.resolve(),
        args.selection.resolve(),
        args.output_root.resolve(),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
