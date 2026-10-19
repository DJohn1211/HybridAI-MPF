"""dataset_pipeline_u_patch.py

Build a *supervision-only* patch dataset for velocity U (Ux, Uy) aligned with an
existing dp_rgh patch dataset produced by `dataset_pipeline_dp_rgh_new.py`.

How it fits your dp_rgh pipeline
-------------------------------
`dataset_pipeline_dp_rgh_new.py` (per case) writes two things side-by-side:

  - out_root/<case>/processed_corrector/<time>.npz   (full-frame, per-time, per-k)
  - out_root/<case>/patch_tXXXX.npz                  (patches, flattened over k)

The *patch_tXXXX.npz* files are the key here: they already encode the exact patch
grid via `coords = [time_index, k, y0, x0]` in a *padded* coordinate system.

To supervise U only (and typically only at the final pressure corrector k=K-1), we:
  1) read each dp patch file to get its patch coords and meta (pad/ps/time_dir)
  2) load the OpenFOAM `U` field in that time folder
  3) apply the *same padding* and then crop U patches at the *same coords*
  4) save `U` patches plus `dp_index` mapping back to dp-file row indices

Output per dp patch file:
  - U: (Nf, 2, ps, ps) for final corrector rows only
  - dp_index: (Nf,) the dp-file row index for each U patch
  - coords: (Nf, 4) same as dp coords subset (time_index,k_final,y0,x0)
  - meta_json: carries through useful meta (time_dir, k_final, etc.)

This lets the training dataset attach U supervision by matching dp row indices.

Important: what is dp_patch_root?
--------------------------------
Set `dp_patch_root` to the *dp_rgh pipeline out_root* (the "total folder") that
contains multiple case subfolders. Each case subfolder contains both
"processed_corrector" and "patch_tXXXX.npz" in the same directory.

This script can also accept `dp_patch_root` pointing directly to a *single case*
directory (the one that contains patch_tXXXX.npz). In that mode, you should set
`single_case_name` in the config if the folder name is not the actual case name.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


_FLOAT = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def _load_meta_json(z: np.lib.npyio.NpzFile) -> Dict:
    """Robustly load meta_json saved by numpy.

    In your dp_rgh pipeline we save: meta_json=json.dumps(meta, ensure_ascii=False)
    which can come back as a 0-d ndarray of dtype '<U...' or as object/bytes.
    """
    if "meta_json" not in z:
        return {}

    raw = z["meta_json"]
    # numpy may return 0-d ndarray for strings
    if isinstance(raw, np.ndarray) and raw.shape == ():
        raw = raw.item()
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    # last resort
    try:
        return json.loads(str(raw))
    except Exception:
        return {}


def _sorted_time_dirs(case_dir: Path) -> List[Path]:
    """OpenFOAM time dirs: numeric folder names, sorted by float value."""
    tdirs: List[Tuple[float, Path]] = []
    for p in case_dir.iterdir():
        if not p.is_dir():
            continue
        try:
            tv = float(p.name)
        except Exception:
            continue
        tdirs.append((tv, p))
    tdirs.sort(key=lambda x: x[0])
    return [p for _, p in tdirs]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_volVectorField_internal(path: Path, n_cells: int) -> np.ndarray:
    """Parse internalField from a volVectorField file.

    Returns: (n_cells, 3) float32
    """
    txt = _read_text(path)

    # uniform
    m = re.search(r"internalField\s+uniform\s*\(\s*(%s)\s+(%s)\s+(%s)\s*\)" % (_FLOAT, _FLOAT, _FLOAT), txt)
    if m:
        v = np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))], dtype=np.float32)
        out = np.repeat(v[None, :], n_cells, axis=0)
        return out

    # nonuniform
    m = re.search(r"internalField\s+nonuniform\s+List<vector>\s*(\d+)\s*\(", txt)
    if not m:
        raise RuntimeError(f"Cannot find internalField in {path}")
    n = int(m.group(1))
    if n != n_cells:
        # allow mismatch, but warn by raising with details
        raise RuntimeError(f"Cell count mismatch for {path}: header n={n}, expected n_cells={n_cells}")

    start = m.end()
    vec_re = re.compile(r"\(\s*(%s)\s+(%s)\s+(%s)\s*\)" % (_FLOAT, _FLOAT, _FLOAT))
    vals = vec_re.findall(txt[start:])
    if len(vals) < n_cells:
        raise RuntimeError(f"Parsed only {len(vals)} vectors from {path}, expected {n_cells}")
    vals = vals[:n_cells]
    arr = np.array(vals, dtype=np.float32)
    return arr


def read_U_2d(time_dir: Path, *, U_name: str, ny: int, nx: int) -> np.ndarray:
    """Return U as (2, ny, nx) float32 (Ux, Uy)."""
    f = time_dir / U_name
    if not f.exists():
        raise FileNotFoundError(f"Missing {U_name} in {time_dir}")
    n_cells = ny * nx
    vec = parse_volVectorField_internal(f, n_cells=n_cells)  # (N,3)
    U = vec.reshape((ny, nx, 3), order="C")
    Ux = U[:, :, 0].astype(np.float32)
    Uy = U[:, :, 1].astype(np.float32)
    return np.stack([Ux, Uy], axis=0)


def safe_pad_2d(arr: np.ndarray, pad_y: int, pad_x: int, *, mode: str = "symmetric") -> np.ndarray:
    """Pad in a way compatible with the dp_rgh patch pipeline.

    The mode must match the pressure dataset metadata. Reflect padding becomes
    invalid when its width is at least the field dimension, so only that mode
    falls back to edge padding for very small domains.
    """
    if pad_y <= 0 and pad_x <= 0:
        return arr

    if arr.ndim == 2:
        H, W = arr.shape
        use_mode = "edge" if mode == "reflect" and ((pad_y >= H and pad_y > 0) or (pad_x >= W and pad_x > 0)) else mode
        return np.pad(arr, ((pad_y, pad_y), (pad_x, pad_x)), mode=use_mode)
    elif arr.ndim == 3:
        C, H, W = arr.shape
        use_mode = "edge" if mode == "reflect" and ((pad_y >= H and pad_y > 0) or (pad_x >= W and pad_x > 0)) else mode
        return np.pad(arr, ((0, 0), (pad_y, pad_y), (pad_x, pad_x)), mode=use_mode)
    else:
        raise ValueError("arr must be 2D or 3D")


def ensure_min_size_2d(arr: np.ndarray, min_h: int, min_w: int, *, mode: str = "edge") -> np.ndarray:
    """Pad (symmetrically) to ensure H>=min_h and W>=min_w."""
    if arr.ndim == 2:
        H, W = arr.shape
        pad_y = max(0, min_h - H)
        pad_x = max(0, min_w - W)
        top = pad_y // 2
        bot = pad_y - top
        left = pad_x // 2
        right = pad_x - left
        return np.pad(arr, ((top, bot), (left, right)), mode=mode)
    elif arr.ndim == 3:
        C, H, W = arr.shape
        pad_y = max(0, min_h - H)
        pad_x = max(0, min_w - W)
        top = pad_y // 2
        bot = pad_y - top
        left = pad_x // 2
        right = pad_x - left
        return np.pad(arr, ((0, 0), (top, bot), (left, right)), mode=mode)
    else:
        raise ValueError("arr must be 2D or 3D")


@dataclass
class UPatchConfig:
    """Build supervision-only U patches aligned with dp_rgh patch coords."""

    # OpenFOAM case root (contains many case subfolders)
    dataset_root: Path

    # dp_rgh pipeline out_root (the total folder).
    # Typical layout (per case):
    #   dp_patch_root/<case>/processed_corrector/
    #   dp_patch_root/<case>/patch_tXXXX.npz
    #
    # You MAY also set dp_patch_root directly to one case folder
    # that contains patch_tXXXX.npz.
    dp_patch_root: Path

    # output root for U patches (if write_alongside_dp=False)
    out_root: Path

    # OpenFOAM field name
    U_name: str = "U"

    # case folder filtering (same default as your dp pipeline)
    case_name_regex: Optional[str] = r"^\d+$"

    # When dp_patch_root points to a single case folder and its folder name is NOT
    # the real case name under dataset_root, set this explicitly.
    single_case_name: Optional[str] = None

    # dp patch file discovery
    dp_patch_glob: str = "patch_t*.npz"
    # if you put patch files in a subdir under each case, set this (usually None)
    dp_patch_dirname: Optional[str] = None

    # which corrector rows to keep:
    #   - "final": keep only k=max(k) rows
    #   - "all":   keep all k rows
    k_mode: str = "final"

    # output placement
    # If True, write U patches next to dp patches (same case folder).
    # In that case you MUST change the filename template to avoid overwriting dp.
    write_alongside_dp: bool = False
    out_filename_template: str = "u_{dp_name}"  # e.g. u_patch_t0000.npz

    # writing behavior
    skip_if_exists: bool = True
    force: bool = False

    # optional: also dump full-frame U per time (debug)
    save_processed_u: bool = False
    processed_u_subdir: str = "processed_U"

    # logging
    verbose: bool = True


def validate_config(cfg: UPatchConfig) -> None:
    cfg.dataset_root = Path(cfg.dataset_root).expanduser()
    cfg.dp_patch_root = Path(cfg.dp_patch_root).expanduser()
    cfg.out_root = Path(cfg.out_root).expanduser()
    if not cfg.dataset_root.is_dir():
        raise FileNotFoundError(f"dataset_root not found: {cfg.dataset_root}")
    if not cfg.dp_patch_root.is_dir():
        raise FileNotFoundError(f"dp_patch_root not found: {cfg.dp_patch_root}")
    if cfg.k_mode not in {"final", "all"}:
        raise ValueError("k_mode must be 'final' or 'all'.")
    if cfg.case_name_regex:
        re.compile(cfg.case_name_regex)


def _match_case_name(name: str, case_name_regex: Optional[str]) -> bool:
    if not case_name_regex:
        return True
    try:
        return re.match(case_name_regex, name) is not None
    except re.error:
        return True


def _resolve_time_dir(case_dir: Path, time_dir_str: str, tmap: Dict[str, Path]) -> Path:
    """Resolve time_dir stored in dp patch meta to a real folder."""
    p = Path(time_dir_str)
    if p.is_absolute() and p.exists():
        return p
    cand = case_dir / time_dir_str
    if cand.exists():
        return cand
    base = p.name
    if base in tmap:
        return tmap[base]
    raise FileNotFoundError(f"Cannot resolve time_dir '{time_dir_str}' under case: {case_dir}")


def _dp_patch_dir(dp_case_dir: Path, cfg: UPatchConfig) -> Path:
    return (dp_case_dir / cfg.dp_patch_dirname) if cfg.dp_patch_dirname else dp_case_dir


def _iter_dp_cases(cfg: UPatchConfig) -> List[Tuple[str, Path]]:
    """Return [(case_name, dp_case_dir), ...] inferred from dp_patch_root."""
    root = Path(cfg.dp_patch_root)
    # Single-case mode: dp_patch_root itself contains patch files
    if root.is_dir() and any(_dp_patch_dir(root, cfg).glob(cfg.dp_patch_glob)):
        case_name = cfg.single_case_name or root.name
        return [(case_name, root)]

    out: List[Tuple[str, Path]] = []
    if not root.is_dir():
        raise FileNotFoundError(f"dp_patch_root not found: {root}")

    for dp_case_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        case_name = dp_case_dir.name
        if not _match_case_name(case_name, cfg.case_name_regex):
            continue
        if any(_dp_patch_dir(dp_case_dir, cfg).glob(cfg.dp_patch_glob)):
            out.append((case_name, dp_case_dir))
    return out


def build_u_patches_for_case(case_name: str, dp_case_dir: Path, cfg: UPatchConfig) -> None:
    case_dir = Path(cfg.dataset_root) / case_name
    if not case_dir.exists():
        raise FileNotFoundError(f"Cannot find case '{case_name}' under dataset_root: {case_dir}")

    dp_case_dir = Path(dp_case_dir)
    dp_patch_dir = _dp_patch_dir(dp_case_dir, cfg)
    dp_files = sorted(dp_patch_dir.glob(cfg.dp_patch_glob))
    if len(dp_files) == 0:
        return

    # where to write
    if cfg.write_alongside_dp:
        out_case_dir = dp_case_dir
    else:
        out_case_dir = Path(cfg.out_root) / case_name
        # Safety: avoid accidentally overwriting dp patches if user points out_root to dp_patch_root
        if out_case_dir.resolve() == dp_case_dir.resolve():
            raise RuntimeError(
                f"out_root ({cfg.out_root}) resolves to the same case folder as dp patches ({dp_case_dir}). "
                "This would overwrite dp patch_tXXXX.npz. "
                "Choose a different out_root, or set write_alongside_dp=True and use out_filename_template."
            )
        out_case_dir.mkdir(parents=True, exist_ok=True)

    time_dirs = _sorted_time_dirs(case_dir)
    tmap: Dict[str, Path] = {p.name: p for p in time_dirs}

    if cfg.save_processed_u:
        (out_case_dir / cfg.processed_u_subdir).mkdir(parents=True, exist_ok=True)

    for dp_path in dp_files:
        # output filename
        if cfg.write_alongside_dp:
            out_name = cfg.out_filename_template.format(dp_name=dp_path.name, dp_stem=dp_path.stem)
            if out_name == dp_path.name:
                raise RuntimeError(
                    f"write_alongside_dp=True but out_filename_template produced '{out_name}', which would overwrite dp patches. "
                    "Use a template like 'u_{dp_name}' or 'u_{dp_stem}.npz'."
                )
        else:
            out_name = dp_path.name
        out_path = out_case_dir / out_name

        if out_path.exists() and cfg.skip_if_exists and (not cfg.force):
            if cfg.verbose:
                print(f"[SKIP] {case_name}/{out_name} -> already exists")
            continue

        with np.load(dp_path, allow_pickle=True) as z:
            coords = z["coords"].astype(np.int64)  # (N,4)
            meta = _load_meta_json(z)
            # optional fallback: infer ps from arrays if meta missing
            X_arr = z["X"] if ("X" in z.files) else None
            Y_arr = z["Y"] if ("Y" in z.files) else None

        time_dir_str = meta.get("time_dir", None)
        if time_dir_str is None:
            raise RuntimeError(
                f"dp patch file missing meta['time_dir']: {dp_path}. "
                "(This field is written by dataset_pipeline_dp_rgh_new.py; please regenerate dp patches.)"
            )
        time_dir = _resolve_time_dir(case_dir, str(time_dir_str), tmap)

        # ny/nx come from dp meta (orig_shape=[ny,nx])
        orig = meta.get("orig_shape", None)
        if isinstance(orig, (list, tuple)) and len(orig) == 2:
            ny, nx = int(orig[0]), int(orig[1])
        else:
            ny, nx = int(meta.get("ny", 0)), int(meta.get("nx", 0))
        if ny <= 0 or nx <= 0:
            raise RuntimeError(f"Bad orig_shape in meta for {dp_path}: {meta.get('orig_shape')}")

        ps = int(meta.get("patch_size", 0))
        if ps <= 0:
            if X_arr is not None:
                ps = int(X_arr.shape[-1])
            elif Y_arr is not None:
                ps = int(Y_arr.shape[-1])
        if ps <= 0:
            raise RuntimeError(f"Cannot infer patch_size from meta or arrays: {dp_path}")

        pad_y = int(meta.get("pad_y", ps // 2))
        pad_x = int(meta.get("pad_x", ps // 2))

        # select rows by k
        ks = coords[:, 1]
        k_final = int(ks.max())
        if cfg.k_mode.lower() == "all":
            sel = np.arange(coords.shape[0], dtype=np.int64)
        else:
            sel = np.where(ks == k_final)[0]
        if sel.size == 0:
            if cfg.verbose:
                print(f"[SKIP] {case_name}/{dp_path.name} -> no rows after k selection")
            continue

        coords_sel = coords[sel]
        dp_index = sel.astype(np.int64)

        # load raw U (2,ny,nx)
        U_raw = read_U_2d(time_dir, U_name=cfg.U_name, ny=ny, nx=nx)

        # optional full-frame debug dump (raw, no padding)
        if cfg.save_processed_u:
            proc_path = out_case_dir / cfg.processed_u_subdir / f"{time_dir.name}.npz"
            if (not proc_path.exists()) or cfg.force:
                np.savez_compressed(
                    proc_path,
                    U=U_raw.astype(np.float32),
                    meta_json=json.dumps({"case": str(case_dir), "time_dir": time_dir.name, "orig_shape": [ny, nx]}, ensure_ascii=False),
                )

        # IMPORTANT: dp coords are in padded frame. We must pad U the same way.
        padding_mode = str(meta.get("physical_padding_mode", "symmetric"))
        if padding_mode not in {"symmetric", "reflect", "edge"}:
            raise RuntimeError(f"Unsupported physical_padding_mode={padding_mode!r} in {dp_path}")
        U = safe_pad_2d(U_raw, pad_y=pad_y, pad_x=pad_x, mode=padding_mode)
        U = ensure_min_size_2d(U, min_h=ps, min_w=ps, mode="edge")

        # crop patches by coords (y0,x0 are in padded frame)
        Up = np.empty((coords_sel.shape[0], 2, ps, ps), dtype=np.float32)
        _, Hpad, Wpad = U.shape
        for i, (_, k, y0, x0) in enumerate(coords_sel):
            y0i = int(y0)
            x0i = int(x0)
            if (y0i < 0) or (x0i < 0) or (y0i + ps > Hpad) or (x0i + ps > Wpad):
                raise RuntimeError(
                    f"Patch coord out of bounds: case={case_name} file={dp_path.name} "
                    f"k={int(k)} y0={y0i} x0={x0i} ps={ps} U_padded=({Hpad},{Wpad}) time_dir={time_dir}"
                )
            Up[i] = U[:, y0i : y0i + ps, x0i : x0i + ps]

        out_meta = dict(meta) if isinstance(meta, dict) else {}
        out_meta.update(
            {
                "k_final": k_final,
                "k_mode": cfg.k_mode,
                "U_name": cfg.U_name,
                "dp_patch_file": dp_path.name,
                "physical_padding_mode": padding_mode,
                "coord_format": meta.get("coord_format", "[time_index, k, y0, x0]"),
            }
        )

        np.savez_compressed(
            out_path,
            U=Up,
            dp_index=dp_index,
            coords=coords_sel,
            meta_json=json.dumps(out_meta, ensure_ascii=False),
        )
        if cfg.verbose:
            print(f"[OK] {case_name}/{dp_path.name} rows={Up.shape[0]} -> {out_path}")


def main(cfg: UPatchConfig) -> None:
    validate_config(cfg)
    cfg.out_root.mkdir(parents=True, exist_ok=True)

    # dp patches define what to build
    cases = _iter_dp_cases(cfg)
    if cfg.verbose:
        print(f"[INFO] dp cases found: {len(cases)}")

    for case_name, dp_case_dir in cases:
        build_u_patches_for_case(case_name, dp_case_dir, cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build U supervision patches aligned with dp_rgh patch coords")
    ap.add_argument("--dataset-root", type=Path, required=True, help="Root containing OpenFOAM cases")
    ap.add_argument("--dp-patch-root", type=Path, required=True, help="dp_rgh pipeline out_root (or one case dir)")
    ap.add_argument("--out-root", type=Path, required=True, help="Output root for U patches")
    ap.add_argument("--u-name", default="U")
    ap.add_argument("--case-name-regex", default=r"^\d+$")
    ap.add_argument("--dp-patch-glob", default="patch_t*.npz")
    ap.add_argument("--dp-patch-dirname")
    ap.add_argument("--single-case-name")
    ap.add_argument("--k-mode", choices=["final", "all"], default="final", help="Keep rows by k")
    ap.add_argument("--write-alongside-dp", action="store_true")
    ap.add_argument("--out-filename-template", default="u_{dp_name}")
    ap.add_argument("--save-processed-u", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    cfg = UPatchConfig(
        dataset_root=args.dataset_root,
        dp_patch_root=args.dp_patch_root,
        out_root=args.out_root,
        U_name=args.u_name,
        case_name_regex=args.case_name_regex,
        dp_patch_glob=args.dp_patch_glob,
        dp_patch_dirname=args.dp_patch_dirname,
        single_case_name=args.single_case_name,
        k_mode=args.k_mode,
        write_alongside_dp=args.write_alongside_dp,
        out_filename_template=args.out_filename_template,
        save_processed_u=args.save_processed_u,
        force=args.force,
        verbose=not args.quiet,
    )
    main(cfg)
