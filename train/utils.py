from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
import torch


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    """Capture the random streams needed for exact checkpoint continuation."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0], numpy_state[1].tolist(), numpy_state[2],
            numpy_state[3], numpy_state[4],
        ),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore random streams saved by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
        numpy_state[2], numpy_state[3], numpy_state[4],
    ))
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_case_dirs(patch_root: Path) -> List[Path]:
    patch_root = patch_root.resolve()
    case_dirs = [d for d in patch_root.iterdir() if d.is_dir()]

    out: List[Path] = []
    for d in case_dirs:
        if any(d.glob("patch_t*.npz")):
            out.append(d)

    # allow root-level npz (single-case)
    if any(patch_root.glob("patch_t*.npz")):
        out.append(patch_root)

    if not out:
        raise FileNotFoundError(
            f"No case directory or root-level patch_t*.npz files found under {patch_root}"
        )

    return sorted(out, key=lambda p: p.name)


def split_cases(
    case_dirs: List[Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> Tuple[List[Path], List[Path], List[Path]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(case_dirs))
    rng.shuffle(idx)

    n = len(idx)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(0, int(round(n * val_ratio)))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    train = [case_dirs[i] for i in idx[:n_train]]
    val = [case_dirs[i] for i in idx[n_train:n_train + n_val]]
    test = [case_dirs[i] for i in idx[n_train + n_val:]]
    return train, val, test


def split_cases_by_ids(
    case_dirs: List[Path],
    train_ids: Sequence[int],
    val_ids: Sequence[int],
    test_ids: Sequence[int],
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Create the fixed case-level split reported in the paper.

    Directory names may be plain integers (``1`` or ``01``) or contain a
    ``case`` token (for example ``Case1`` or ``case_01_bubble``).  Ambiguous,
    duplicate, missing, or unexpected case identifiers are rejected so a
    nominally reproducible run cannot silently use the wrong simulations.
    """

    def case_id(path: Path) -> int:
        name = path.name
        if name.isdigit():
            return int(name)
        match = re.search(r"(?i)(?:^|[^a-z])case[_ -]*0*(\d+)(?:[^0-9]|$)", name)
        if match is None:
            raise ValueError(
                f"Cannot infer the paper case number from directory {name!r}. "
                "Rename it to Case<N> or use an explicit supported case name."
            )
        return int(match.group(1))

    indexed = {}
    for path in case_dirs:
        cid = case_id(path)
        if cid in indexed:
            raise ValueError(
                f"Duplicate case id {cid}: {indexed[cid].name!r} and {path.name!r}"
            )
        indexed[cid] = path

    train_ids = tuple(int(v) for v in train_ids)
    val_ids = tuple(int(v) for v in val_ids)
    test_ids = tuple(int(v) for v in test_ids)
    expected = set(train_ids) | set(val_ids) | set(test_ids)
    if len(expected) != len(train_ids) + len(val_ids) + len(test_ids):
        raise ValueError("train/validation/test case id lists must not overlap")

    missing = sorted(expected - set(indexed))
    unexpected = sorted(set(indexed) - expected)
    if missing or unexpected:
        raise ValueError(
            "Dataset cases do not match the paper split. "
            f"Missing ids={missing}; unexpected ids={unexpected}."
        )

    return (
        [indexed[cid] for cid in train_ids],
        [indexed[cid] for cid in val_ids],
        [indexed[cid] for cid in test_ids],
    )
