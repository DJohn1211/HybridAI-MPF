"""
OpenFOAM interFoam (pressure corrector level) dataset pipeline

Goal：
    X^k = [alpha, rhs_corrk, rAU_corrk, p_in, p_rgh_in, invDx2, invDy2, x, y, mask]
    Y^k = Δp_rgh^k = p_rgh_corrk - p_rgh_in
keys：
  - k=0: p_rgh_in from last time step final p_rgh (prev_time/p_rgh)
  - k>0: p_rgh_in = current time step p_rgh_corr(k-1)
  - p_in ：p_in = p_rgh_in + (p - p_rgh)_{current time step}
    where (p - p_rgh) = rho*gh

Output：
  Part A: processed_corrector/<time>.npz
    - X: (K, C, ny, nx)
    - Y: (K, 1, ny, nx)
  Part B: patches: out_root/<case>/patch_tXXXX.npz
    - X: (K*Np, C, ps, ps)
    - Y: (K*Np, 1, ps, ps)
    - coords: (K*Np, 4) = [time_index, k, y0, x0]
"""

from __future__ import annotations
import argparse
import os
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ======================================================================
# Part A: OpenFOAM reading utilities (copied/adapted from your previous pipeline)
# ======================================================================

def _read_text(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()


def infer_mesh_and_spacing(blockMesh_path: str):
    """从 blockMeshDict 推断 nx, ny, nz 以及尺度信息（默认取第一个 hex）。"""
    with open(blockMesh_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    scale = 1.0
    m_scale = re.search(r"scale\s+([0-9eE\+\-\.]+)\s*;", text)
    if m_scale:
        scale = float(m_scale.group(1))

    m_vertices = re.search(r"vertices\s*\((.*?)\);", text, re.S)
    if not m_vertices:
        raise RuntimeError(f"{blockMesh_path}: 未找到 vertices 段")

    vertices_block = m_vertices.group(1)
    vert_pattern = r"\(\s*([0-9eE\+\-\.]+)\s+([0-9eE\+\-\.]+)\s+([0-9eE\+\-\.]+)\s*\)"
    verts = re.findall(vert_pattern, vertices_block)
    xs = [float(v[0]) for v in verts]
    ys = [float(v[1]) for v in verts]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x0 = x_min * scale
    y0 = y_min * scale
    Lx = (x_max - x_min) * scale
    Ly = (y_max - y_min) * scale

    pattern_hex = r"hex\s*\([^)]*\)\s*\((\d+)\s+(\d+)\s+(\d+)\)"
    matches = re.findall(pattern_hex, text)
    if not matches:
        raise RuntimeError(f"{blockMesh_path}: 未找到 hex (...) (Nx Ny Nz) 定义")
    nx, ny, nz = map(int, matches[0])

    dx = Lx / max(nx, 1)
    dy = Ly / max(ny, 1)
    return nx, ny, nz, dx, dy, x0, y0, Lx, Ly, scale


def _find_internalfield_anchor(lines: List[str]) -> int:
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        if "internalField" in s:
            return i
    return -1


def _scan_uniform_value(lines: List[str], start: int) -> Optional[str]:
    """从 start 行起向后扫描几行，找 uniform value 段"""
    for j in range(start, min(start + 8, len(lines))):
        s = lines[j].strip()
        if not s or s.startswith("//"):
            continue
        if "uniform" in s:
            return s
    return None


def parse_scalar_field(field_path: str, n_cells_expected: int) -> np.ndarray:
    lines = _read_text(field_path)
    anchor = _find_internalfield_anchor(lines)
    if anchor < 0:
        raise RuntimeError(f"{field_path}: 未找到 internalField")

    # uniform scalar
    uline = _scan_uniform_value(lines, anchor)
    if uline and "uniform" in uline and "nonuniform" not in uline:
        m = re.search(r"uniform\s+([0-9eE\+\-\.]+)\s*;", uline)
        if not m:
            parts = uline.replace(";", " ").split()
            idx = parts.index("uniform")
            val = float(parts[idx + 1])
        else:
            val = float(m.group(1))
        return np.full((n_cells_expected,), val, dtype=np.float32)

    # nonuniform
    nonu = -1
    for j in range(anchor, min(anchor + 10, len(lines))):
        if "nonuniform" in lines[j]:
            nonu = j
            break
    if nonu < 0:
        raise RuntimeError(f"{field_path}: 未找到 internalField nonuniform（也不是 uniform）")

    j = nonu + 1
    while j < len(lines) and (not lines[j].strip() or lines[j].strip().startswith("//")):
        j += 1
    if j < len(lines) and "List" in lines[j]:
        j += 1
        while j < len(lines) and (not lines[j].strip() or lines[j].strip().startswith("//")):
            j += 1
    n_cells = int(lines[j].strip())

    k = j + 1
    while k < len(lines) and "(" not in lines[k]:
        k += 1
    idx = k + 1

    vals: List[float] = []
    for l in range(idx, min(idx + n_cells, len(lines))):
        val = lines[l].strip().strip("()")
        if val:
            vals.append(float(val))

    if len(vals) != n_cells:
        raise RuntimeError(f"{field_path}: 读取标量数={len(vals)} 与 nCells={n_cells} 不一致")
    if n_cells_expected and n_cells != n_cells_expected:
        print(f"[WARN] {field_path}: nCells={n_cells} != expected={n_cells_expected}")

    return np.array(vals, dtype=np.float32)


def find_time_dirs(case_root: str, skip_time0: bool = True, time0_eps: float = 1e-12) -> List[str]:
    """收集 case_root 下的时间步目录名（可选跳过 time=0）。"""
    dirs: List[str] = []
    for name in os.listdir(case_root):
        full = os.path.join(case_root, name)
        if not os.path.isdir(full):
            continue
        if name in ("system", "constant"):
            continue
        try:
            t = float(name)
        except ValueError:
            continue
        if skip_time0 and abs(t) <= time0_eps:
            continue
        dirs.append(name)
    return sorted(dirs, key=lambda x: float(x))


def _find_corr_indices(time_path: str, base: str) -> List[int]:
    """
    在 time_path 下找形如 f"{base}_corr0" 的文件，返回可用 k 列表（升序）。
    """
    ks = []
    pat = re.compile(rf"^{re.escape(base)}_corr(\d+)$")
    for fn in os.listdir(time_path):
        m = pat.match(fn)
        if m:
            ks.append(int(m.group(1)))
    return sorted(set(ks))


def _contiguous_prefix(ks: List[int]) -> List[int]:
    """把 [0,1,2,4] 截成 [0,1,2]，并保证从0开始。"""
    if not ks or ks[0] != 0:
        return []
    out = [0]
    for k in ks[1:]:
        if k == out[-1] + 1:
            out.append(k)
        else:
            break
    return out


# ======================================================================
# Part A: build processed (per time step, per corrector k)
# ======================================================================

def process_case_corrector(
    case_root: str,
    out_dir: str,
    alpha_name: str = "alpha.water",
    p_name: str = "p",
    p_rgh_name: str = "p_rgh",
    rhs_base: str = "rhs",
    rAU_base: str = "rAU",
    prgh_corr_base: str = "p_rgh",
    skip_time0: bool = True,
    verbose: bool = True,
) -> None:
    """
    为一个 case 生成 processed_corrector/*.npz（每个时间步一份，包含 K 个 corrector 样本）

    依赖当前时间步目录内存在：
      - alpha.water, p, p_rgh
      - rhs_corrk, rAU_corrk, p_rgh_corrk  (k=0..K-1)
    """
    os.makedirs(out_dir, exist_ok=True)
    time_dirs = find_time_dirs(case_root, skip_time0=skip_time0)
    if len(time_dirs) < 2:
        raise RuntimeError(f"{case_root}: Need >=2 time dirs after skip_time0={skip_time0}")

    blockMesh_path = os.path.join(case_root, "system", "blockMeshDict")
    nx, ny, nz, dx, dy, x0, y0, Lx, Ly, scale = infer_mesh_and_spacing(blockMesh_path)
    if int(nz) != 1:
        raise RuntimeError(f"{case_root}: assume 2D nz=1, got nz={nz}")
    n_cells_expected = int(nx * ny * nz)

    # 静态通道：x,y（归一化到[0,1]），invDx2/invDy2
    xc = x0 + (np.arange(nx, dtype=np.float32) + 0.5) * dx
    yc = y0 + (np.arange(ny, dtype=np.float32) + 0.5) * dy
    x = ((xc[None, :] - x0) / (Lx + 1e-12)).astype(np.float32)
    y = ((yc[:, None] - y0) / (Ly + 1e-12)).astype(np.float32)
    x = np.broadcast_to(x, (ny, nx)).copy()
    y = np.broadcast_to(y, (ny, nx)).copy()
    invDx2 = np.full((ny, nx), 1.0 / (dx * dx), dtype=np.float32)
    invDy2 = np.full((ny, nx), 1.0 / (dy * dy), dtype=np.float32)

    # 维护上一时间步最终 p_rgh（用于当前时间步 k=0 输入）
    prev_prgh_final: Optional[np.ndarray] = None
    wrote = 0

    for ti, tdir in enumerate(time_dirs):
        time_path = os.path.join(case_root, tdir)

        a_path     = os.path.join(time_path, alpha_name)
        p_path     = os.path.join(time_path, p_name)
        prgh_path  = os.path.join(time_path, p_rgh_name)

        if not (os.path.isfile(a_path) and os.path.isfile(p_path) and os.path.isfile(prgh_path)):
            if verbose:
                print(f"[SKIP] {Path(case_root).name}/{tdir}: missing alpha/p/p_rgh")
            continue

        # 先确定 K：交集（rhs_corrk, rAU_corrk, p_rgh_corrk）且从0连续
        ks_rhs  = _find_corr_indices(time_path, rhs_base)
        ks_rAU  = _find_corr_indices(time_path, rAU_base)
        ks_prgh = _find_corr_indices(time_path, prgh_corr_base)  # p_rgh_corrk

        ks = sorted(set(ks_rhs).intersection(ks_rAU).intersection(ks_prgh))
        ks = _contiguous_prefix(ks)
        if not ks:
            if verbose:
                print(f"[SKIP] {Path(case_root).name}/{tdir}: no contiguous corr files (rhs/rAU/p_rgh)")
            # 仍然更新 prev_prgh_final，保证后面能跑
            try:
                prgh_final_1d = parse_scalar_field(prgh_path, n_cells_expected)
                prev_prgh_final = prgh_final_1d.reshape(ny, nx).astype(np.float32)
            except Exception:
                pass
            continue

        # 读当前时间步的基础场
        try:
            alpha_1d = parse_scalar_field(a_path, n_cells_expected)
            p_1d     = parse_scalar_field(p_path, n_cells_expected)
            pr_1d    = parse_scalar_field(prgh_path, n_cells_expected)
        except Exception as e:
            if verbose:
                print(f"[SKIP] {Path(case_root).name}/{tdir}: parse base fields failed -> {e}")
            continue

        alpha = alpha_1d.reshape(ny, nx).astype(np.float32)
        p     = p_1d.reshape(ny, nx).astype(np.float32)
        p_rgh = pr_1d.reshape(ny, nx).astype(np.float32)

        # rho*gh（当前时间步）= p - p_rgh
        rhoGh = (p - p_rgh).astype(np.float32)

        # 第一个可用时间步：没有 prev_prgh_final（无法做 k=0），只更新 prev 然后跳过写样本
        if prev_prgh_final is None:
            prev_prgh_final = p_rgh.copy()
            if verbose:
                print(f"[INIT] prev_prgh_final set by time {tdir}, skip dataset for this time.")
            continue

        K = len(ks)
        C = 10
        X = np.zeros((K, C, ny, nx), dtype=np.float32)
        Y = np.zeros((K, 1, ny, nx), dtype=np.float32)

        # 逐 k 构造样本
        prgh_in = prev_prgh_final  # k=0 input
        for k in range(K):
            rhs_path_k  = os.path.join(time_path, f"{rhs_base}_corr{k}")
            rAU_path_k  = os.path.join(time_path, f"{rAU_base}_corr{k}")
            prgh_out_kp = os.path.join(time_path, f"{prgh_corr_base}_corr{k}")  # p_rgh_corrk

            try:
                rhs_1d   = parse_scalar_field(rhs_path_k, n_cells_expected)
                rAU_1d   = parse_scalar_field(rAU_path_k, n_cells_expected)
                prout_1d = parse_scalar_field(prgh_out_kp, n_cells_expected)
            except Exception as e:
                raise RuntimeError(f"{Path(case_root).name}/{tdir}: read corr{k} failed -> {e}")

            rhs   = rhs_1d.reshape(ny, nx).astype(np.float32)
            rAU   = rAU_1d.reshape(ny, nx).astype(np.float32)
            prout = prout_1d.reshape(ny, nx).astype(np.float32)

            # 用物理定义得到 p_in（保证与 prgh_in 自洽）
            p_in = (prgh_in + rhoGh).astype(np.float32)

            mask = np.ones((ny, nx), dtype=np.float32)

            # X^k = [alpha, rhs, rAU, p_in, p_rgh_in, invDx2, invDy2, x, y, mask]
            X[k, 0] = alpha
            X[k, 1] = rhs
            X[k, 2] = rAU
            X[k, 3] = p_in
            X[k, 4] = prgh_in
            X[k, 5] = invDx2
            X[k, 6] = invDy2
            X[k, 7] = x
            X[k, 8] = y
            X[k, 9] = mask

            # Y^k = Δp_rgh = p_rgh_out - p_rgh_in
            Y[k, 0] = (prout - prgh_in).astype(np.float32)

            # 下一轮输入来自本轮输出
            prgh_in = prout

        meta = dict(
            time_dir=str(tdir),
            time_index=int(ti),
            nx=int(nx), ny=int(ny), nz=int(nz),
            dx=float(dx), dy=float(dy), x0=float(x0), y0=float(y0), Lx=float(Lx), Ly=float(Ly), scale=float(scale),
            K=int(K),
            # IMPORTANT: X[3] is p_in (constructed), X[4] is p_rgh_in (autoregressive state)
            in_channels=["alpha", "rhs", "rAU", "p_in", "p_rgh_in", "invDx2", "invDy2", "x", "y", "mask"],
            out_channels=["delta_p_rgh"],
            note="k=pressure corrector index; k=0 uses prev time final p_rgh as input; p_in = p_rgh_in + (p - p_rgh)@current time",
        )

        out_path = os.path.join(out_dir, f"{tdir}.npz")
        np.savez_compressed(out_path, X=X, Y=Y, meta_json=json.dumps(meta, ensure_ascii=False))
        wrote += 1
        if verbose:
            print(f"[OK] processed_corrector wrote {out_path}  (K={K})")

        # 更新 prev（用于下一时间步 k=0）
        prev_prgh_final = p_rgh.copy()

    if wrote < 1:
        raise RuntimeError(f"{case_root}: wrote 0 processed_corrector npz, check your corr outputs.")


# ======================================================================
# Part B: patch utils (copied/adapted)
# ======================================================================

def _time_key(npz_path: Path) -> float:
    stem = npz_path.stem
    try:
        return float(stem)
    except ValueError:
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", stem)
        return float(m.group(1)) if m else float("inf")


def list_npz(processed_dir: Path) -> List[Path]:
    files = sorted(processed_dir.glob("*.npz"), key=_time_key)
    if len(files) < 1:
        raise RuntimeError(f"Need >= 1 npz files in: {processed_dir}")
    return files


def compute_stride(patch_size: int, overlap_ratio: float) -> int:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError("overlap_ratio must be in [0,1).")
    return max(1, int(round(patch_size * (1.0 - overlap_ratio))))


def sliding_positions(size: int, patch_size: int, stride: int) -> List[int]:
    if size < patch_size:
        raise ValueError(f"size={size} is smaller than patch_size={patch_size}.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    pos = list(range(0, size - patch_size + 1, stride))
    if not pos:
        return [0]
    if pos[-1] != size - patch_size:
        pos.append(size - patch_size)
    return pos


def safe_pad_2d(
    a: np.ndarray,
    pad_y: int,
    pad_x: int,
    mode: str = "symmetric",
) -> np.ndarray:
    """Pad a physical field using the configured boundary extension."""
    if pad_y <= 0 and pad_x <= 0:
        return a.astype(np.float32, copy=False)
    if mode not in {"symmetric", "reflect", "edge"}:
        raise ValueError(f"Unsupported physical padding mode: {mode}")
    H, W = a.shape
    use_mode = mode
    if mode == "reflect" and ((pad_y >= H and pad_y > 0) or (pad_x >= W and pad_x > 0)):
        use_mode = "edge"
    return np.pad(a, ((pad_y, pad_y), (pad_x, pad_x)), mode=use_mode).astype(np.float32)


def pad_coord_2d(a: np.ndarray, pad_y: int, pad_x: int) -> np.ndarray:
    """Coordinate-like padding: use edge replication (no reflection).

    Why: reflect padding makes x/y non-monotone near the boundary, which is
    physically weird and can confuse checks and models. Edge padding keeps
    x/y piecewise monotone (flat in the padded halo).
    """
    if pad_y <= 0 and pad_x <= 0:
        return a.astype(np.float32, copy=False)
    return np.pad(a, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge").astype(np.float32)


def pad_mask_2d(mask: np.ndarray, pad_y: int, pad_x: int) -> np.ndarray:
    if pad_y <= 0 and pad_x <= 0:
        return mask.astype(np.float32, copy=False)
    return np.pad(mask, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant", constant_values=0.0).astype(np.float32)


def ensure_min_size_2d(a: np.ndarray, min_h: int, min_w: int, mode: str) -> np.ndarray:
    H, W = a.shape
    add_y = max(0, min_h - H)
    add_x = max(0, min_w - W)
    if add_y == 0 and add_x == 0:
        return a.astype(np.float32, copy=False)
    y0 = add_y // 2
    y1 = add_y - y0
    x0 = add_x // 2
    x1 = add_x - x0
    if mode == "constant0":
        return np.pad(a, ((y0, y1), (x0, x1)), mode="constant", constant_values=0.0).astype(np.float32)
    return np.pad(a, ((y0, y1), (x0, x1)), mode="edge").astype(np.float32)


def crop_patches(frame: np.ndarray, patch_size: int, overlap_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    C, H, W = frame.shape
    if patch_size > H or patch_size > W:
        raise ValueError(f"patch_size={patch_size} > frame size=({H},{W}); increase padding.")
    stride = compute_stride(patch_size, overlap_ratio)
    ys = sliding_positions(H, patch_size, stride)
    xs = sliding_positions(W, patch_size, stride)

    patches = np.empty((len(ys) * len(xs), C, patch_size, patch_size), dtype=np.float32)
    coords = np.empty((len(ys) * len(xs), 2), dtype=np.int32)

    k = 0
    for y0 in ys:
        for x0 in xs:
            patches[k] = frame[:, y0:y0 + patch_size, x0:x0 + patch_size]
            coords[k] = (y0, x0)
            k += 1
    return patches, coords


def load_processed(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    with np.load(npz_path, allow_pickle=True) as z:
        X = z["X"].astype(np.float32)  # (K,C,H,W)
        Y = z["Y"].astype(np.float32)  # (K,1,H,W)
        meta = {}
        if "meta_json" in z:
            try:
                meta = json.loads(str(z["meta_json"]))
            except Exception:
                meta = {}
    return X, Y, meta


# ======================================================================
# Pipeline config + running
# ======================================================================

@dataclass
class Config:
    dataset_root: Path
    out_root: Path
    case_name_regex: Optional[str] = r"^\d+$"

    # Part A
    enable_preprocess: bool = True
    processed_subdir: str = "processed_corrector"
    processed_in_out_root: bool = True  # 输出到 out_root/<case>/<processed_subdir>/
    force_preprocess: bool = False
    skip_time0_in_preprocess: bool = True

    alpha_name: str = "alpha.water"
    p_name: str = "p"
    p_rgh_name: str = "p_rgh"
    rhs_base: str = "rhs"
    rAU_base: str = "rAU"
    prgh_corr_base: str = "p_rgh"

    # Part B
    enable_patch: bool = True
    patch_size: int = 64
    overlap_ratio: float = 0.5
    pad_x: Optional[int] = None
    pad_y: Optional[int] = None
    physical_padding_mode: str = "symmetric"
    compress: bool = True
    force_patch: bool = False
    continue_on_error: bool = False
    verbose: bool = True


EXPECTED_IN_CHANNELS = [
    "alpha", "rhs", "rAU", "p_in", "p_rgh_in",
    "invDx2", "invDy2", "x", "y", "mask",
]


def validate_config(cfg: Config) -> None:
    cfg.dataset_root = Path(cfg.dataset_root).expanduser()
    cfg.out_root = Path(cfg.out_root).expanduser()
    if not cfg.dataset_root.is_dir():
        raise FileNotFoundError(f"dataset_root not found: {cfg.dataset_root}")
    if cfg.patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    compute_stride(cfg.patch_size, cfg.overlap_ratio)
    if cfg.pad_x is not None and cfg.pad_x < 0:
        raise ValueError("pad_x must be non-negative.")
    if cfg.pad_y is not None and cfg.pad_y < 0:
        raise ValueError("pad_y must be non-negative.")
    if cfg.physical_padding_mode not in {"symmetric", "reflect", "edge"}:
        raise ValueError("physical_padding_mode must be symmetric, reflect, or edge.")
    if cfg.case_name_regex:
        re.compile(cfg.case_name_regex)


def find_cases(cfg: Config) -> List[Path]:
    if not cfg.dataset_root.is_dir():
        raise RuntimeError(f"dataset_root not found: {cfg.dataset_root}")
    pat = re.compile(cfg.case_name_regex) if cfg.case_name_regex else None
    cases: List[Path] = []
    for p in cfg.dataset_root.iterdir():
        if not p.is_dir():
            continue
        if pat and (pat.match(p.name) is None):
            continue
        cases.append(p)
    cases.sort(key=lambda x: x.name)
    if not cases:
        raise RuntimeError(
            f"No case directories matched {cfg.case_name_regex!r} under {cfg.dataset_root}"
        )
    return cases


def maybe_preprocess_case(case_dir: Path, cfg: Config) -> Path:
    processed_dir = (cfg.out_root / case_dir.name / cfg.processed_subdir) if cfg.processed_in_out_root else (case_dir / cfg.processed_subdir)
    has_npz = processed_dir.is_dir() and any(processed_dir.glob("*.npz"))

    if cfg.enable_preprocess:
        if cfg.force_preprocess or (not has_npz):
            processed_dir.mkdir(parents=True, exist_ok=True)
            if cfg.verbose:
                print(f"[preprocess_corrector] {case_dir.name} -> {processed_dir}")
            process_case_corrector(
                str(case_dir),
                str(processed_dir),
                alpha_name=cfg.alpha_name,
                p_name=cfg.p_name,
                p_rgh_name=cfg.p_rgh_name,
                rhs_base=cfg.rhs_base,
                rAU_base=cfg.rAU_base,
                prgh_corr_base=cfg.prgh_corr_base,
                skip_time0=cfg.skip_time0_in_preprocess,
                verbose=cfg.verbose,
            )
        else:
            if cfg.verbose:
                print(f"[preprocess_corrector] {case_dir.name} skipped (found npz in {processed_dir})")
    else:
        if not has_npz:
            raise RuntimeError(f"{case_dir.name}: processed_corrector npz not found. Enable preprocess or generate first.")
    return processed_dir


def build_patches_for_case(case_dir: Path, processed_dir: Path, cfg: Config) -> None:
    files = list_npz(processed_dir)

    ps = int(cfg.patch_size)
    pad_x = int(ps // 2) if cfg.pad_x is None else int(cfg.pad_x)
    pad_y = int(ps // 2) if cfg.pad_y is None else int(cfg.pad_y)
    save_fn = np.savez_compressed if cfg.compress else np.savez

    out_case_dir = cfg.out_root / case_dir.name
    out_case_dir.mkdir(parents=True, exist_ok=True)

    for time_i, f in enumerate(files):
        Xk, Yk, meta = load_processed(f)  # Xk:(K,C,H,W), Yk:(K,1,H,W)
        K, C, ny, nx = Xk.shape
        if C != len(EXPECTED_IN_CHANNELS):
            raise RuntimeError(f"{f}: expected 10 input channels, got {C}")
        if Yk.shape != (K, 1, ny, nx):
            raise RuntimeError(f"{f}: invalid Y shape {Yk.shape}; expected {(K, 1, ny, nx)}")
        in_names = meta.get("in_channels", EXPECTED_IN_CHANNELS)
        if list(in_names) != EXPECTED_IN_CHANNELS:
            raise RuntimeError(f"{f}: unexpected channel order: {in_names}")

        out_path = out_case_dir / f"patch_t{time_i:04d}.npz"
        if out_path.exists() and not cfg.force_patch:
            if cfg.verbose:
                print(f"[SKIP] {out_path} already exists (use --force-patch to rebuild)")
            continue

        X_all = []
        Y_all = []
        coords_all = []

        for k in range(K):
            X = Xk[k]        # (C,ny,nx)
            Y = Yk[k]        # (1,ny,nx)

            # padding + min size（mask 用 constant0 padding）
            # Default to our canonical channel naming (p_in / p_rgh_in) in case meta is missing.
            Xp_ch = []
            for ci, nm in enumerate(in_names):
                if nm == "mask":
                    Xp_ch.append(pad_mask_2d(X[ci], pad_y, pad_x))
                elif nm in ("x", "y", "invDx2", "invDy2"):
                    # x/y/invDx2/invDy2 are coordinate-like; avoid reflect padding
                    Xp_ch.append(pad_coord_2d(X[ci], pad_y, pad_x))
                else:
                    Xp_ch.append(safe_pad_2d(X[ci], pad_y, pad_x, cfg.physical_padding_mode))
            Xp = np.stack(Xp_ch, axis=0).astype(np.float32)
            Yp = np.stack(
                [safe_pad_2d(Y[0], pad_y, pad_x, cfg.physical_padding_mode)], axis=0
            ).astype(np.float32)

            Xp2 = []
            for ci, nm in enumerate(in_names):
                if nm == "mask":
                    Xp2.append(ensure_min_size_2d(Xp[ci], ps, ps, mode="constant0"))
                else:
                    Xp2.append(ensure_min_size_2d(Xp[ci], ps, ps, mode="edge"))
            Xp = np.stack(Xp2, axis=0).astype(np.float32)
            Yp = np.stack([ensure_min_size_2d(Yp[0], ps, ps, mode="edge")], axis=0).astype(np.float32)

            Xpatch, coords = crop_patches(Xp, ps, cfg.overlap_ratio)
            Ypatch, coords2 = crop_patches(Yp, ps, cfg.overlap_ratio)
            if coords.shape[0] != coords2.shape[0] or not np.all(coords == coords2):
                raise RuntimeError("X/Y coords mismatch")

            # coords: [time_index, k, y0, x0]
            coords4 = np.concatenate(
                [
                    np.full((coords.shape[0], 1), time_i, dtype=np.int32),
                    np.full((coords.shape[0], 1), k, dtype=np.int32),
                    coords.astype(np.int32),
                ],
                axis=1,
            )

            X_all.append(Xpatch)
            Y_all.append(Ypatch)
            coords_all.append(coords4)

        X_all = np.concatenate(X_all, axis=0)
        Y_all = np.concatenate(Y_all, axis=0)
        coords_all = np.concatenate(coords_all, axis=0)

        meta2 = dict(meta) if isinstance(meta, dict) else {}
        meta2.update({
            "case": case_dir.name,
            "processed_file": f"{cfg.processed_subdir}/{f.name}",
            "patch_size": ps,
            "overlap_ratio": float(cfg.overlap_ratio),
            "stride": int(compute_stride(ps, cfg.overlap_ratio)),
            "pad_x": int(pad_x),
            "pad_y": int(pad_y),
            "orig_shape": [int(ny), int(nx)],
            "time_dir": str(f.stem),
            "coord_format": "[time_index, k, y0, x0]",
            "physical_padding_mode": cfg.physical_padding_mode,
            "coordinate_padding_mode": "edge",
            "mask_padding_mode": "constant0",
            "in_channels": EXPECTED_IN_CHANNELS,
            "out_channels": ["delta_p_rgh"],
        })

        save_fn(out_path, X=X_all, Y=Y_all, coords=coords_all, meta_json=json.dumps(meta2, ensure_ascii=False))
        if cfg.verbose:
            print(f"[patch] {case_dir.name} time={time_i:04d} K={K} N={X_all.shape[0]} -> {out_path}")


def run_pipeline(cfg: Config) -> None:
    validate_config(cfg)
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    cases = find_cases(cfg)
    errors: List[str] = []
    for case_dir in cases:
        try:
            processed_dir = maybe_preprocess_case(case_dir, cfg)
            if cfg.enable_patch:
                build_patches_for_case(case_dir, processed_dir, cfg)
        except Exception as e:
            message = f"{case_dir.name}: {e}"
            errors.append(message)
            print(f"[ERROR] {message}")
            if not cfg.continue_on_error:
                raise

    manifest = {
        "format_version": 1,
        "pipeline": "pressure_corrector_delta_p_rgh",
        "cases": [p.name for p in cases],
        "failed_cases": errors,
        "patch_size": cfg.patch_size,
        "overlap_ratio": cfg.overlap_ratio,
        "stride": compute_stride(cfg.patch_size, cfg.overlap_ratio),
        "pad_x": cfg.patch_size // 2 if cfg.pad_x is None else cfg.pad_x,
        "pad_y": cfg.patch_size // 2 if cfg.pad_y is None else cfg.pad_y,
        "physical_padding_mode": cfg.physical_padding_mode,
        "in_channels": EXPECTED_IN_CHANNELS,
        "out_channels": ["delta_p_rgh"],
    }
    (cfg.out_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if errors:
        raise RuntimeError(f"Dataset generation completed with {len(errors)} failed case(s).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the pressure-corrector delta-p_rgh patch dataset."
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root containing OpenFOAM case directories.")
    parser.add_argument("--out-root", type=Path, required=True, help="Output dataset directory.")
    parser.add_argument("--case-name-regex", default=r"^\d+$")
    parser.add_argument("--patch-size", type=int, default=64, help="Paper configuration: 64.")
    parser.add_argument("--overlap-ratio", type=float, default=0.5, help="Paper configuration: 0.5.")
    parser.add_argument("--pad-x", type=int)
    parser.add_argument("--pad-y", type=int)
    parser.add_argument(
        "--physical-padding-mode",
        choices=["symmetric", "reflect", "edge"],
        default="symmetric",
    )
    parser.add_argument("--no-preprocess", action="store_true", help="Use existing processed_corrector files.")
    parser.add_argument("--no-patches", action="store_true", help="Only generate processed_corrector files.")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--force-patch", action="store_true")
    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument("--include-time-zero", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = Config(
        dataset_root=args.dataset_root,
        out_root=args.out_root,
        case_name_regex=args.case_name_regex,
        enable_preprocess=not args.no_preprocess,
        enable_patch=not args.no_patches,
        force_preprocess=args.force_preprocess,
        force_patch=args.force_patch,
        skip_time0_in_preprocess=not args.include_time_zero,
        patch_size=args.patch_size,
        overlap_ratio=args.overlap_ratio,
        pad_x=args.pad_x,
        pad_y=args.pad_y,
        physical_padding_mode=args.physical_padding_mode,
        compress=not args.no_compress,
        continue_on_error=args.continue_on_error,
        verbose=not args.quiet,
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
