from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ----------------------------
# basic utils
# ----------------------------
def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _parse_meta(meta_any) -> Dict:
    if meta_any is None:
        return {}
    try:
        return json.loads(str(meta_any))
    except Exception:
        return {}


def _finite_stats(a: np.ndarray) -> Dict[str, float]:
    fin = np.isfinite(a)
    if not fin.any():
        return {"finite_ratio": 0.0}
    v = a[fin].astype(np.float64)
    return {
        "finite_ratio": float(fin.mean()),
        "min": float(v.min()),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "std": float(v.std()),
    }


def _rel_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    num = float(np.sqrt(np.mean((a - b) ** 2)))
    den = float(np.sqrt(np.mean(b ** 2)) + eps)
    return num / den


def compute_stride(patch_size: int, overlap_ratio: float) -> int:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("overlap_ratio must be in [0, 1)")
    return max(1, int(round(patch_size * (1.0 - overlap_ratio))))


def sliding_positions(size: int, patch_size: int, stride: int) -> List[int]:
    if size < patch_size:
        raise ValueError(f"size={size} is smaller than patch_size={patch_size}")
    if stride <= 0:
        raise ValueError("stride must be positive")
    pos = list(range(0, size - patch_size + 1, stride))
    if not pos:
        return [0]
    if pos[-1] != size - patch_size:
        pos.append(size - patch_size)
    return pos


# ----------------------------
# Poisson-ish discrete operator
#   approximates: div( rAU * grad(p) )
#   (cell-centered rAU, face avg)
# ----------------------------
def apply_div_rAU_grad(
    p: np.ndarray,
    rAU: np.ndarray,
    invDx2: np.ndarray,
    invDy2: np.ndarray,
) -> np.ndarray:
    """
    p,rAU,invDx2,invDy2: (H,W), float32/64
    Returns: (H,W)
    Notes:
      - This is a structured-grid approximation, not bitwise OpenFOAM.
      - It is still very good at catching channel mis-order, alignment bugs,
        and gross physical inconsistency.
    """
    p = p.astype(np.float64)
    rAU = rAU.astype(np.float64)
    invDx2 = invDx2.astype(np.float64)
    invDy2 = invDy2.astype(np.float64)

    H, W = p.shape
    out = np.zeros((H, W), dtype=np.float64)

    # x-direction faces (between j and j+1)
    if W >= 2:
        rAUf_x = 0.5 * (rAU[:, 1:] + rAU[:, :-1])       # (H, W-1)
        dp_x = (p[:, 1:] - p[:, :-1])                   # (H, W-1)
        flux_x = rAUf_x * dp_x                          # missing /dx, handled by invDx2 later

        # interior
        if W > 2:
            out[:, 1:-1] += (flux_x[:, 1:] - flux_x[:, :-1]) * invDx2[:, 1:-1]
        # boundaries (one-sided, effectively zero-grad-ish)
        out[:, 0] += (flux_x[:, 0]) * invDx2[:, 0]
        out[:, -1] += (-flux_x[:, -1]) * invDx2[:, -1]

    # y-direction faces (between i and i+1)
    if H >= 2:
        rAUf_y = 0.5 * (rAU[1:, :] + rAU[:-1, :])       # (H-1, W)
        dp_y = (p[1:, :] - p[:-1, :])                   # (H-1, W)
        flux_y = rAUf_y * dp_y

        if H > 2:
            out[1:-1, :] += (flux_y[1:, :] - flux_y[:-1, :]) * invDy2[1:-1, :]
        out[0, :] += (flux_y[0, :]) * invDy2[0, :]
        out[-1, :] += (-flux_y[-1, :]) * invDy2[-1, :]

    return out.astype(np.float32)


def poisson_residual_metrics(
    p: np.ndarray,
    rhs: np.ndarray,
    rAU: np.ndarray,
    invDx2: np.ndarray,
    invDy2: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Returns two variants because rhs sign conventions differ across exports:
      resA = div(rAU grad p) - rhs
      resB = div(rAU grad p) + rhs
    We report relL2 for both; the smaller one is the more plausible convention.
    """
    A = apply_div_rAU_grad(p, rAU, invDx2, invDy2).astype(np.float32)
    rhs = rhs.astype(np.float32)

    if mask is not None:
        m = mask.astype(np.float32)
        # avoid all-zero mask
        w = np.maximum(m, 0.0)
        if float(w.mean()) > 1e-6:
            A = A * w
            rhs = rhs * w

    # compute relL2 with rhs magnitude as reference
    relA = _rel_l2(A, rhs)
    relB = _rel_l2(A, -rhs)

    # also absolute scales
    fin = np.isfinite(A) & np.isfinite(rhs)
    if fin.any():
        rmse = float(np.sqrt(np.mean((A[fin] - rhs[fin]) ** 2)))
        rhs_rms = float(np.sqrt(np.mean(rhs[fin] ** 2)) + 1e-12)
    else:
        rmse, rhs_rms = float("nan"), float("nan")

    return {
        "relL2_divGrad_minus_rhs": relA,
        "relL2_divGrad_plus_rhs": relB,
        "rmse_minus_rhs": rmse,
        "rhs_rms": rhs_rms,
    }


# ----------------------------
# dataset structure checks
# ----------------------------
def resolve_processed_file(patch_path: Path, meta: Dict, processed_subdir: str = "processed_corrector") -> Optional[Path]:
    """
    Robustly find processed npz even if meta['processed_file'] is stale.
    Priority:
      1) meta['processed_file'] if exists
      2) <patch_dir>/processed_corrector/<time_dir>.npz using meta['time_dir']
      3) infer by time_index (coords[:,0]) and sorted processed list
    """
    # 1) absolute path in meta
    p0 = meta.get("processed_file", None)
    if p0:
        p0 = Path(str(p0))
        if p0.exists():
            return p0

        # common rename: underscore <-> dash
        p0_alt = Path(str(p0).replace("dp_rgh_patch", "dp_rgh-patch"))
        if p0_alt.exists():
            return p0_alt

    # 2) relative to patch_dir + time_dir
    time_dir = meta.get("time_dir", None)
    if time_dir:
        cand = patch_path.parent / processed_subdir / f"{time_dir}.npz"
        if cand.exists():
            return cand

    # 3) infer from coords time_index
    #    patch file lives in <out_root>/<case>/patch_tXXXX.npz, and processed are next to it
    proc_dir = patch_path.parent / processed_subdir
    if proc_dir.is_dir():
        files = sorted(proc_dir.glob("*.npz"), key=lambda x: _float_stem(x.stem))
        if files:
            # patch index itself equals time_index used in coords & meta
            m = re.search(r"patch_t(\d+)", patch_path.stem)
            if m:
                idx = int(m.group(1))
                if 0 <= idx < len(files):
                    return files[idx]
    return None


def _float_stem(stem: str) -> float:
    try:
        return float(stem)
    except ValueError:
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", stem)
        return float(m.group(1)) if m else float("inf")


def check_grid_channels(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    x,y: (H,W)
    Checks: within [0,1] and "looks like coordinates".
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    H, W = x.shape

    # basic range
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))

    # coordinate-likeness:
    # x should vary mostly along columns, y mostly along rows.
    # measure negative diff ratio (a strong detector of reflect-padding artifacts on coords)
    dx = np.diff(x, axis=1) if W > 1 else np.zeros((H, 0))
    dy = np.diff(y, axis=0) if H > 1 else np.zeros((0, W))
    neg_dx_ratio = float((dx < -1e-6).mean()) if dx.size else 0.0
    neg_dy_ratio = float((dy < -1e-6).mean()) if dy.size else 0.0

    # x should be almost constant along rows for a structured rect grid
    x_row_std = float(np.mean(np.std(x, axis=0))) if W > 1 else 0.0
    y_col_std = float(np.mean(np.std(y, axis=1))) if H > 1 else 0.0

    return {
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "neg_dx_ratio": neg_dx_ratio,
        "neg_dy_ratio": neg_dy_ratio,
        "x_row_std_mean": x_row_std,
        "y_col_std_mean": y_col_std,
    }


def expected_coords_set(meta: Dict) -> Optional[set]:
    """
    From meta2 in patch file:
      orig_shape, pad_x, pad_y, patch_size, overlap_ratio/stride
    We compute the expected (y0,x0) positions in padded frame.
    """
    try:
        H0, W0 = meta["orig_shape"]
        ps = int(meta["patch_size"])
        pad_x = int(meta["pad_x"])
        pad_y = int(meta["pad_y"])
        stride = int(meta.get("stride", compute_stride(ps, float(meta.get("overlap_ratio", 0.0)))))
    except Exception:
        return None

    H = int(H0) + 2 * pad_y
    W = int(W0) + 2 * pad_x
    H = max(H, ps)
    W = max(W, ps)

    ys = sliding_positions(H, ps, stride)
    xs = sliding_positions(W, ps, stride)
    s = set()
    for y0 in ys:
        for x0 in xs:
            s.add((int(y0), int(x0)))
    return s


def check_patch_file(patch_path: Path, processed_subdir: str = "processed_corrector", verbose: bool = True) -> Dict:
    z = _load_npz(patch_path)
    X = z.get("X", None)
    Y = z.get("Y", None)
    coords = z.get("coords", None)
    meta = _parse_meta(z.get("meta_json", None))

    out: Dict = {
        "patch_file": str(patch_path),
        "ok": True,
        "reason": "",
    }

    if X is None or Y is None or coords is None:
        out["ok"] = False
        out["reason"] = "missing X/Y/coords in patch npz"
        return out

    # shapes
    N, C, ps1, ps2 = X.shape
    out.update({"N": int(N), "C": int(C), "ps": int(ps1)})
    if ps1 != ps2:
        out["ok"] = False
        out["reason"] = f"ps not square: {ps1}x{ps2}"
        return out
    if Y.shape[0] != N:
        out["ok"] = False
        out["reason"] = "Y N mismatch"
        return out
    if coords.shape[0] != N or coords.shape[1] != 4:
        out["ok"] = False
        out["reason"] = "coords shape mismatch"
        return out

    # finite check
    out["X_finite"] = float(np.isfinite(X).mean())
    out["Y_finite"] = float(np.isfinite(Y).mean())
    if out["X_finite"] < 1.0 or out["Y_finite"] < 1.0:
        out["ok"] = False
        out["reason"] = "X or Y contains NaN/Inf"
        return out
    if C != 10:
        out["ok"] = False
        out["reason"] = f"expected 10 input channels, got {C}"
        return out

    # infer K / Np from coords
    k_vals = np.unique(coords[:, 1].astype(np.int32))
    K = int(len(k_vals))
    out["K"] = K
    if K <= 0:
        out["ok"] = False
        out["reason"] = "K<=0"
        return out
    if N % K != 0:
        out["ok"] = False
        out["reason"] = f"N%K!=0 (N={N},K={K})"
        return out
    out["Np_per_k"] = int(N // K)

    # time_index consistency: patch_tXXXX should match coords[:,0]
    m = re.search(r"patch_t(\d+)", patch_path.stem)
    if m:
        time_i = int(m.group(1))
        ti_vals = np.unique(coords[:, 0].astype(np.int32))
        out["time_index_unique"] = [int(x) for x in ti_vals[:10]]
        if len(ti_vals) != 1 or int(ti_vals[0]) != time_i:
            out["ok"] = False
            out["reason"] = f"time_index mismatch: filename={time_i}, coords_unique={ti_vals}"
            return out

    # patch coords integrity: compare expected (y0,x0) set per k
    exp = expected_coords_set(meta)
    if exp is not None:
        for k in k_vals:
            sel = coords[:, 1] == k
            got = set(map(tuple, coords[sel][:, 2:4].astype(np.int32)))
            if got != exp:
                out["ok"] = False
                out["reason"] = f"coords set mismatch at k={int(k)} (got={len(got)} exp={len(exp)})"
                return out

    # coordinate-channel check (channels fixed by your pipeline)
    # X channels: [alpha,rhs,rAU,p_in,p_rgh_in,invDx2,invDy2,x,y,mask]  (C=10)
    if C >= 10:
        xch = X[0, 7]
        ych = X[0, 8]
        grid_stat = check_grid_channels(xch, ych)
        out.update({f"grid_{k}": v for k, v in grid_stat.items()})

        # strong warning detector: if coords were reflect-padded, neg_dx_ratio/neg_dy_ratio will rise
        # In a correct "edge padded coordinates" world, these should be ~0.
        out["coord_reflect_like"] = bool(grid_stat["neg_dx_ratio"] > 0.02 or grid_stat["neg_dy_ratio"] > 0.02)

    # resolve processed file & run deeper checks
    proc_path = resolve_processed_file(patch_path, meta, processed_subdir=processed_subdir)
    out["processed_file_resolved"] = str(proc_path) if proc_path else ""
    if proc_path is None or (not proc_path.exists()):
        out["processed_found"] = False
        return out
    out["processed_found"] = True

    deep = check_processed_file(proc_path)
    out.update({f"proc_{k}": v for k, v in deep.items()})

    # corrector alignment check using processed
    align = corrector_alignment_check(proc_path)
    out.update({f"align_{k}": v for k, v in align.items()})

    # physical check (Poisson-ish residuals) on processed
    phys = physical_consistency_check(proc_path)
    out.update({f"phys_{k}": v for k, v in phys.items()})

    if verbose and out.get("coord_reflect_like", False):
        print(f"[WARN] {patch_path.name}: x/y look reflect-padded (neg diff ratio > 2%). "
              f"建议：x/y 通道 padding 用 edge，不要 reflect。")

    return out


def check_processed_file(proc_path: Path) -> Dict:
    z = _load_npz(proc_path)
    Xk = z.get("X", None)  # (K,C,H,W)
    Yk = z.get("Y", None)  # (K,1,H,W)
    meta = _parse_meta(z.get("meta_json", None))

    out: Dict = {"ok": True, "K": -1, "C": -1, "H": -1, "W": -1}
    if Xk is None or Yk is None:
        out["ok"] = False
        out["reason"] = "missing X/Y in processed npz"
        return out

    K, C, H, W = Xk.shape
    out.update({"K": int(K), "C": int(C), "H": int(H), "W": int(W)})
    out["X_finite"] = float(np.isfinite(Xk).mean())
    out["Y_finite"] = float(np.isfinite(Yk).mean())

    # channel sanity (only if C>=10)
    if C >= 10:
        alpha = Xk[:, 0]
        rAU = Xk[:, 2]
        invDx2 = Xk[:, 5]
        invDy2 = Xk[:, 6]
        x = Xk[:, 7]
        y = Xk[:, 8]
        mask = Xk[:, 9]

        out["alpha_out_of_0_1_ratio"] = float(((alpha < -1e-3) | (alpha > 1 + 1e-3)).mean())
        out["rAU_nonpos_ratio"] = float((rAU <= 0).mean())
        out["invDx2_nonpos_ratio"] = float((invDx2 <= 0).mean())
        out["invDy2_nonpos_ratio"] = float((invDy2 <= 0).mean())
        out["mask_not_0_1_ratio"] = float(((mask != 0.0) & (mask != 1.0)).mean())

        # grid checks on k=0
        grid_stat = check_grid_channels(x[0], y[0])
        out.update({f"grid_{k}": v for k, v in grid_stat.items()})

        # invDx2/invDy2 constant check
        out["invDx2_std"] = float(np.std(invDx2[0]))
        out["invDy2_std"] = float(np.std(invDy2[0]))

        # reconstruct dx/dy
        dx = 1.0 / math.sqrt(float(np.mean(invDx2[0])) + 1e-30)
        dy = 1.0 / math.sqrt(float(np.mean(invDy2[0])) + 1e-30)
        out["dx_from_invDx2"] = float(dx)
        out["dy_from_invDy2"] = float(dy)

        # optional cross-check with meta dx/dy if present
        if isinstance(meta, dict) and ("dx" in meta) and ("dy" in meta):
            out["dx_meta"] = float(meta["dx"])
            out["dy_meta"] = float(meta["dy"])
            out["dx_rel_err"] = float(abs(dx - float(meta["dx"])) / (abs(float(meta["dx"])) + 1e-12))
            out["dy_rel_err"] = float(abs(dy - float(meta["dy"])) / (abs(float(meta["dy"])) + 1e-12))

    return out


def corrector_alignment_check(proc_path: Path) -> Dict[str, float]:
    """
    Checks within a time step:
      1) p_rgh_in(k+1) == p_rgh_in(k) + Δp(k)
      2) rhoGh := p_in - p_rgh_in should be invariant across k
    """
    z = _load_npz(proc_path)
    Xk = z["X"].astype(np.float32)  # (K,C,H,W)
    Yk = z["Y"].astype(np.float32)  # (K,1,H,W)
    K, C, H, W = Xk.shape

    if C < 5:
        return {"ok": 0.0, "reason_code": 1.0}

    pr_in = Xk[:, 4]      # (K,H,W)
    dp = Yk[:, 0]         # (K,H,W)
    pr_out = pr_in + dp

    # 1) chain consistency
    if K >= 2:
        chain_err = np.abs(pr_in[1:] - pr_out[:-1])
        chain_rel = float(np.sqrt(np.mean(chain_err ** 2)) / (np.sqrt(np.mean(pr_in[1:] ** 2)) + 1e-12))
    else:
        chain_rel = 0.0

    # 2) rhoGh invariance across k
    if C >= 4:
        rhoGh = Xk[:, 3] - Xk[:, 4]   # (K,H,W)
        rhoGh_ref = rhoGh[0]
        rhoGh_err = rhoGh - rhoGh_ref[None]
        rhoGh_rel = float(np.sqrt(np.mean(rhoGh_err ** 2)) / (np.sqrt(np.mean(rhoGh_ref ** 2)) + 1e-12))
    else:
        rhoGh_rel = 0.0

    return {
        "chain_rel_rmse": chain_rel,
        "rhoGh_rel_drift": rhoGh_rel,
    }


def physical_consistency_check(proc_path: Path) -> Dict[str, float]:
    """
    For each k:
      - compute residual metrics for p_in (p_rgh_in) and p_out (p_rgh_in + Δp)
      - report mean over k
    """
    z = _load_npz(proc_path)
    Xk = z["X"].astype(np.float32)  # (K,C,H,W)
    Yk = z["Y"].astype(np.float32)  # (K,1,H,W)
    K, C, H, W = Xk.shape

    if C < 10:
        return {"ok": 0.0, "reason_code": 1.0}

    rel_in_A, rel_out_A = [], []
    rel_in_B, rel_out_B = [], []
    for k in range(K):
        rhs = Xk[k, 1]
        rAU = Xk[k, 2]
        pr_in = Xk[k, 4]
        pr_out = pr_in + Yk[k, 0]
        invDx2 = Xk[k, 5]
        invDy2 = Xk[k, 6]
        mask = Xk[k, 9]

        m_in = poisson_residual_metrics(pr_in, rhs, rAU, invDx2, invDy2, mask=mask)
        m_out = poisson_residual_metrics(pr_out, rhs, rAU, invDx2, invDy2, mask=mask)

        rel_in_A.append(m_in["relL2_divGrad_minus_rhs"])
        rel_out_A.append(m_out["relL2_divGrad_minus_rhs"])
        rel_in_B.append(m_in["relL2_divGrad_plus_rhs"])
        rel_out_B.append(m_out["relL2_divGrad_plus_rhs"])

    # choose convention by which is smaller on average for p_out
    mean_out_A = float(np.mean(rel_out_A)) if rel_out_A else float("nan")
    mean_out_B = float(np.mean(rel_out_B)) if rel_out_B else float("nan")
    best = "minus_rhs" if mean_out_A <= mean_out_B else "plus_rhs"

    return {
        "best_rhs_convention": 0.0 if best == "minus_rhs" else 1.0,  # numeric flag for logging
        "mean_relL2_in_minus_rhs": float(np.mean(rel_in_A)) if rel_in_A else float("nan"),
        "mean_relL2_out_minus_rhs": float(np.mean(rel_out_A)) if rel_out_A else float("nan"),
        "mean_relL2_in_plus_rhs": float(np.mean(rel_in_B)) if rel_in_B else float("nan"),
        "mean_relL2_out_plus_rhs": float(np.mean(rel_out_B)) if rel_out_B else float("nan"),
    }


# ----------------------------
# scan runner
# ----------------------------
@dataclass
class SanityConfig:
    patch_root: Path
    processed_subdir: str = "processed_corrector"

    # scan control
    max_cases: int = 999999
    max_patch_files_per_case: int = 999999
    verbose: bool = True

    # "usable K" threshold (for reporting)
    min_usable_K: int = 1
    report_json: Optional[Path] = None


def list_cases(patch_root: Path) -> List[Path]:
    return sorted([p for p in patch_root.iterdir() if p.is_dir()], key=lambda p: p.name)


def list_patch_files(case_dir: Path) -> List[Path]:
    return sorted(case_dir.glob("patch_t*.npz"), key=lambda p: p.name)


def run_checks(cfg: SanityConfig) -> Dict:
    if not cfg.patch_root.is_dir():
        raise RuntimeError(f"patch_root not found: {cfg.patch_root}")

    cases = list_cases(cfg.patch_root)[: cfg.max_cases]
    K_hist: Dict[int, int] = {}
    missing_processed = 0
    coord_reflect_like = 0
    total_patch_files = 0
    bad_patch_files = 0
    usableK_files = 0
    file_results: List[Dict] = []

    for ci, case_dir in enumerate(cases):
        patch_files = list_patch_files(case_dir)[: cfg.max_patch_files_per_case]
        if not patch_files:
            continue

        if cfg.verbose:
            print("=" * 80)
            print(f"[case] {case_dir.name}  patch_files={len(patch_files)}")

        for pf in patch_files:
            total_patch_files += 1
            res = check_patch_file(pf, processed_subdir=cfg.processed_subdir, verbose=cfg.verbose)
            file_results.append(res)

            if not res.get("ok", False):
                bad_patch_files += 1
                if cfg.verbose:
                    print(f"[BAD ] {case_dir.name}/{pf.name}: {res.get('reason', '')}")
                continue

            K = int(res.get("K", -1))
            K_hist[K] = K_hist.get(K, 0) + 1
            if K >= cfg.min_usable_K:
                usableK_files += 1

            if not res.get("processed_found", False):
                missing_processed += 1

            if res.get("coord_reflect_like", False):
                coord_reflect_like += 1

            if cfg.verbose:
                msg = (f"[OK ] {case_dir.name}/{pf.name}: "
                       f"N={res['N']} C={res['C']} ps={res['ps']} K={res['K']} "
                       f"proc={'Y' if res.get('processed_found', False) else 'N'}")
                print(msg)

                # quick high-signal diagnostics
                if res.get("processed_found", False):
                    chain = res.get("align_chain_rel_rmse", float("nan"))
                    drift = res.get("align_rhoGh_rel_drift", float("nan"))
                    best_flag = res.get("phys_best_rhs_convention", float("nan"))  # 0.0 => minus_rhs, 1.0 => plus_rhs

                    phys_in_m = res.get("phys_mean_relL2_in_minus_rhs", float("nan"))
                    phys_out_m = res.get("phys_mean_relL2_out_minus_rhs", float("nan"))
                    phys_in_p = res.get("phys_mean_relL2_in_plus_rhs", float("nan"))
                    phys_out_p = res.get("phys_mean_relL2_out_plus_rhs", float("nan"))

                    best_name = "minus_rhs" if best_flag == 0.0 else ("plus_rhs" if best_flag == 1.0 else "unknown")

                    print(f"      align: chain_rel_rmse={chain:.3e}  rhoGh_drift={drift:.3e}")
                    print(f"      phys : relL2_in  (minus_rhs)={phys_in_m:.3e}  (plus_rhs)={phys_in_p:.3e}")  # 解前残差（对应 p_rgh_in vs 当前步 rhs/rAU）
                    print(f"             relL2_out (minus_rhs)={phys_out_m:.3e}  (plus_rhs)={phys_out_p:.3e}  best={best_name}") # 解后残差（对应 p_rgh_out = p_rgh_in + Δp vs 当前步 rhs/rAU）

    print("\n" + "=" * 80)
    print(f"[done] total_patch_files={total_patch_files} bad_patch_files={bad_patch_files}")
    print(f"       missing_processed={missing_processed}")
    print(f"       coord_reflect_like_warnings={coord_reflect_like}")
    print(f"       usableK_files(K>={cfg.min_usable_K})={usableK_files}")
    print("       K_hist:", dict(sorted(K_hist.items(), key=lambda kv: kv[0])))

    # hard suggestion
    if coord_reflect_like > 0:
        print("\n[SUGGEST] 检测到不少 patch 的 x/y 通道呈现 reflect-padding 痕迹（负差分比例>2%）。")
        print("          强烈建议：x/y 坐标通道 padding 用 edge（常值外延），不要用 reflect。")

    summary = {
        "ok": bad_patch_files == 0 and total_patch_files > 0,
        "total_patch_files": total_patch_files,
        "bad_patch_files": bad_patch_files,
        "missing_processed": missing_processed,
        "coord_reflect_like_warnings": coord_reflect_like,
        "usable_K_files": usableK_files,
        "K_hist": dict(sorted(K_hist.items())),
        "files": file_results,
    }
    if cfg.report_json is not None:
        cfg.report_json.parent.mkdir(parents=True, exist_ok=True)
        cfg.report_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate generated delta-p_rgh patch datasets.")
    parser.add_argument("--patch-root", type=Path, required=True)
    parser.add_argument("--processed-subdir", default="processed_corrector")
    parser.add_argument("--max-cases", type=int, default=999999)
    parser.add_argument("--max-patch-files-per-case", type=int, default=999999)
    parser.add_argument("--min-usable-k", type=int, default=1)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_checks(SanityConfig(
        patch_root=args.patch_root,
        processed_subdir=args.processed_subdir,
        max_cases=args.max_cases,
        max_patch_files_per_case=args.max_patch_files_per_case,
        min_usable_K=args.min_usable_k,
        report_json=args.report_json,
        verbose=not args.quiet,
    ))
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
