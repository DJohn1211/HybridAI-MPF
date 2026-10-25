"""Validate that pressure and velocity patch datasets are file-aligned."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def check_alignment(dp_root: Path, u_root: Path, sample_files: int | None) -> int:
    if not dp_root.is_dir():
        raise FileNotFoundError(f"Pressure-patch root not found: {dp_root}")
    if not u_root.is_dir():
        raise FileNotFoundError(f"Velocity-patch root not found: {u_root}")

    missing_cases: list[str] = []
    missing_files: list[tuple[str, str]] = []
    invalid_files: list[tuple[str, str, str]] = []
    checked_files = 0
    found_cases = 0

    for case_dir in sorted((p for p in dp_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        dp_files = sorted(case_dir.glob("patch_t*.npz"))
        if not dp_files:
            continue
        found_cases += 1

        u_case = u_root / case_dir.name
        if not u_case.is_dir():
            missing_cases.append(case_dir.name)
            continue

        files_to_check = dp_files if sample_files is None else dp_files[:sample_files]
        for dp_file in files_to_check:
            checked_files += 1
            u_file = u_case / dp_file.name
            if not u_file.is_file():
                missing_files.append((case_dir.name, dp_file.name))
                continue
            try:
                with np.load(dp_file, allow_pickle=False) as dp_data:
                    missing_keys = {"X", "Y", "coords"} - set(dp_data.files)
                    if missing_keys:
                        raise ValueError(f"pressure NPZ missing keys {sorted(missing_keys)}")
                    x_shape = dp_data["X"].shape
                    y_shape = dp_data["Y"].shape
                    coords_shape = dp_data["coords"].shape
                    pressure_coords = dp_data["coords"]
                if len(x_shape) != 4 or x_shape[1:] != (10, 64, 64):
                    raise ValueError(f"X shape must be (N,10,64,64), got {x_shape}")
                if len(y_shape) != 4 or y_shape[0] != x_shape[0] or y_shape[1:] != (1, 64, 64):
                    raise ValueError(f"Y shape must be (N,1,64,64) and align with X, got {y_shape}")
                if len(coords_shape) != 2 or coords_shape != (x_shape[0], 4):
                    raise ValueError(f"coords shape must be (N,4), got {coords_shape}")
                if not np.issubdtype(pressure_coords.dtype, np.integer):
                    raise ValueError("pressure coords must contain integers")
                reference = pressure_coords[pressure_coords[:, 1] == 0, 2:4]
                if reference.shape[0] == 0:
                    raise ValueError("corrector k=0 is missing")
                for corrector in (1, 2):
                    positions = pressure_coords[pressure_coords[:, 1] == corrector, 2:4]
                    if positions.shape[0] != reference.shape[0] or set(map(tuple, positions)) != set(map(tuple, reference)):
                        raise ValueError(f"corrector k={corrector} is incomplete for the paper's three-step rollout")

                with np.load(u_file, allow_pickle=False) as u_data:
                    if "U" not in u_data.files:
                        raise ValueError("velocity NPZ missing key 'U'")
                    u_shape = u_data["U"].shape
                    dp_index = u_data["dp_index"] if "dp_index" in u_data.files else None
                    velocity_coords = u_data["coords"] if "coords" in u_data.files else None
                if len(u_shape) != 4 or u_shape[1:] != (2, 64, 64):
                    raise ValueError(f"U shape must be (Nu,2,64,64), got {u_shape}")
                if u_shape[0] != x_shape[0]:
                    if dp_index is None or dp_index.shape != (u_shape[0],):
                        raise ValueError(
                            "subset velocity data require dp_index with one entry per U row"
                        )
                if dp_index is None:
                    if u_shape[0] != x_shape[0]:
                        raise ValueError("subset velocity data require dp_index")
                    dp_index = np.arange(x_shape[0], dtype=np.int64)
                if dp_index.shape != (u_shape[0],) or not np.issubdtype(dp_index.dtype, np.integer):
                    raise ValueError("dp_index must contain one integer per U row")
                if dp_index.size == 0 or dp_index.min() < 0 or dp_index.max() >= x_shape[0]:
                    raise ValueError("dp_index contains out-of-range pressure-row indices")
                if np.unique(dp_index).size != dp_index.size:
                    raise ValueError("dp_index contains duplicate pressure-row indices")
                if velocity_coords is not None and not np.array_equal(velocity_coords, pressure_coords[dp_index]):
                    raise ValueError("velocity coords do not match pressure coords at dp_index")
                required_velocity_rows = set(np.flatnonzero(pressure_coords[:, 1] == 2))
                if not required_velocity_rows.issubset(set(dp_index.tolist())):
                    raise ValueError("velocity rows for k=2 are missing; build U patches with --k-mode all")
            except Exception as exc:
                invalid_files.append((case_dir.name, dp_file.name, str(exc)))

    print(f"Checked {checked_files} pressure-patch files.")
    if found_cases == 0:
        print("No case directories containing patch_t*.npz were found.")
        return 1
    if missing_cases:
        print(f"Cases missing from velocity root: {missing_cases}")
    if missing_files:
        print(f"Missing aligned velocity files: {missing_files[:20]}")
        if len(missing_files) > 20:
            print(f"... and {len(missing_files) - 20} more")
    if invalid_files:
        print("Invalid aligned files:")
        for case_name, file_name, reason in invalid_files[:20]:
            print(f"  {case_name}/{file_name}: {reason}")
        if len(invalid_files) > 20:
            print(f"... and {len(invalid_files) - 20} more")

    if missing_cases or missing_files or invalid_files:
        return 1
    print("Dataset alignment check passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-root", type=Path, required=True)
    parser.add_argument("--u-patch-root", type=Path, required=True)
    parser.add_argument(
        "--sample-files",
        type=int,
        default=None,
        help="Check only the first N files per case; default checks every file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sample_files is not None and args.sample_files <= 0:
        raise ValueError("--sample-files must be positive")
    raise SystemExit(check_alignment(args.patch_root, args.u_patch_root, args.sample_files))


if __name__ == "__main__":
    main()
