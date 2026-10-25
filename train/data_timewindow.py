from __future__ import annotations

"""Load pressure patches into physical-time or corrector-index windows.

For the paper's corrector mode, pressure NPZ files contain X with ten channels
[alpha, rhs, rAU, p_in, p_rgh_in, invDx2, invDy2, x, y, mask], Y with
delta_p_rgh, and coords with [time_index, corrector_index, y0, x0]. The last
three X channels are the normalized x/y coordinates and mask. The iterable
returns physical history, future fields, targets, and geometry for rollout;
the trainer may replace pressure channels under scheduled teacher forcing.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


# =========================
# NPZ cache
# =========================


class NPZCache:
    """Small cache for patch_tXXXX.npz.

    Supports:
      - time-window datasets: X, Y, meta_json
      - corrector datasets: X, Y, coords, meta_json  (coords=[time_index,k,y0,x0])
    """

    def __init__(self, max_items: int = 8):
        self.max_items = int(max_items)
        self._keys: List[Path] = []
        # value: (X, Y, coords_or_None, meta_dict)
        self._vals: Dict[Path, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict]] = {}

    def get_full(self, path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict]:
        """Return (X, Y, coords_or_None, meta_dict) for one npz."""
        path = Path(path)
        if path in self._vals:
            X, Y, coords, meta = self._vals[path]
            return X, Y, coords, meta

        with np.load(path, allow_pickle=False) as z:
            X = z["X"]
            Y = z["Y"]
            coords = z["coords"] if "coords" in z.files else None

            meta: Dict = {}
            if "meta_json" in z.files:
                try:
                    raw = z["meta_json"].item() if hasattr(z["meta_json"], "item") else z["meta_json"]
                    meta = json.loads(raw)
                except Exception:
                    meta = {}

        if len(self._keys) >= self.max_items:
            old = self._keys.pop(0)
            self._vals.pop(old, None)

        self._keys.append(path)
        self._vals[path] = (X, Y, coords, meta)
        return X, Y, coords, meta

    def get(self, path: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Backward-compatible: return (X, Y, meta_dict)."""
        X, Y, _coords, meta = self.get_full(path)
        return X, Y, meta


# =========================
# Normalization stats
# =========================

@dataclass
class NormStats:
    x_mean: np.ndarray  # (n_phys,)
    x_std: np.ndarray   # (n_phys,)
    y_mean: float
    y_std: float

    def to_json(self) -> Dict:
        return {
            "x_mean": self.x_mean.tolist(),
            "x_std": self.x_std.tolist(),
            "y_mean": float(self.y_mean),
            "y_std": float(self.y_std),
        }

    @staticmethod
    def from_json(d: Dict) -> "NormStats":
        return NormStats(
            x_mean=np.asarray(d["x_mean"], dtype=np.float32),
            x_std=np.asarray(d["x_std"], dtype=np.float32),
            y_mean=float(d["y_mean"]),
            y_std=float(d["y_std"]),
        )


def save_json(path: Path, d: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict:
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

    return sorted(out)


def _accum_masked_stats(sum_: np.ndarray, sumsq: np.ndarray, count: float, data: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    sum_/sumsq: (C,)
    data: (B,C,H,W)
    mask: (B,1,H,W) in {0,1}
    """
    w = mask.astype(np.float32)
    # (B,C,H,W)
    dw = data.astype(np.float32) * w
    sum_ += dw.sum(axis=(0, 2, 3))
    sumsq += (dw * data.astype(np.float32)).sum(axis=(0, 2, 3))
    count += float(w.sum())
    return sum_, sumsq, count


def compute_norm_stats(
    case_dirs: List[Path],
    *,
    max_files_per_case: int = 6,
    sample_patches_per_file: int = 128,
    seed: int = 0,
) -> NormStats:
    """
    Compute per-channel mean/std for ALL physical channels (X without last 3 grid channels),
    using mask-weighted pixels.

    y stats are computed from Y[:,0:1] with the same mask from X.
    """
    rng = np.random.default_rng(int(seed))
    cache = NPZCache(max_items=4)

    n_phys: Optional[int] = None
    sum_x: Optional[np.ndarray] = None
    sumsq_x: Optional[np.ndarray] = None
    cnt_x: float = 0.0

    sum_y: float = 0.0
    sumsq_y: float = 0.0
    cnt_y: float = 0.0

    for case_dir in case_dirs:
        case_dir = Path(case_dir)
        files = sorted(case_dir.glob("patch_t*.npz"))
        if not files:
            continue

        pick = files if len(files) <= max_files_per_case else list(rng.choice(files, size=max_files_per_case, replace=False))
        for fp in pick:
            X, Y, _ = cache.get(fp)
            if X.ndim != 4:
                raise RuntimeError(f"{fp}: X must be 4D (Np,C,H,W), got {X.shape}")
            if X.shape[1] < 4:
                raise RuntimeError(f"{fp}: X channels too small: C={X.shape[1]}")
            if n_phys is None:
                n_phys = int(X.shape[1] - 3)
                sum_x = np.zeros((n_phys,), dtype=np.float64)
                sumsq_x = np.zeros((n_phys,), dtype=np.float64)
            else:
                if int(X.shape[1] - 3) != n_phys:
                    raise RuntimeError(f"{fp}: inconsistent n_phys across files: {X.shape[1]-3} vs {n_phys}")

            Np = X.shape[0]
            if sample_patches_per_file >= Np:
                pidx = np.arange(Np, dtype=np.int64)
            else:
                pidx = rng.choice(Np, size=sample_patches_per_file, replace=False)

            Xs = X[pidx].astype(np.float32)
            phys = Xs[:, :n_phys, :, :]
            grid = Xs[:, -3:, :, :]
            mask = grid[:, 2:3, :, :]

            sum_x, sumsq_x, cnt_x = _accum_masked_stats(sum_x, sumsq_x, cnt_x, phys, mask)

            Ys = Y[pidx].astype(np.float32)
            y0 = Ys[:, 0:1, :, :]
            w = mask.astype(np.float32)
            sum_y += float((y0 * w).sum())
            sumsq_y += float(((y0 * y0) * w).sum())
            cnt_y += float(w.sum())

    if n_phys is None or sum_x is None or sumsq_x is None or cnt_x <= 0:
        raise RuntimeError("compute_norm_stats: no valid samples found. Check patch_root/case_dirs.")

    mean_x = (sum_x / cnt_x).astype(np.float32)
    var_x = (sumsq_x / cnt_x - mean_x.astype(np.float64) ** 2).clip(min=1e-12)
    std_x = np.sqrt(var_x).astype(np.float32)

    y_mean = sum_y / max(cnt_y, 1e-12)
    y_var = max(sumsq_y / max(cnt_y, 1e-12) - y_mean * y_mean, 1e-12)
    y_std = float(np.sqrt(y_var))

    return NormStats(x_mean=mean_x, x_std=std_x, y_mean=float(y_mean), y_std=float(y_std))


# =========================
# Batch + Iterable dataset
# =========================

@dataclass
class Batch:
    # physical history: (B,T_in,n_phys,H,W)
    phys_hist: torch.Tensor
    # future truth frames: (B,T_out-1,n_phys,H,W)
    phys_fut: torch.Tensor
    # targets: delta_p for T_out steps: (B,T_out,1,H,W)
    dp_fut: torch.Tensor
    # grid_norm: (B,3,H,W) = [x_norm,y_norm,mask]  (this is what the MODEL sees)
    grid: torch.Tensor
    # grid_phys: (B,3,H,W) = [x_phys,y_phys,mask]  (this is what PDE/gh/dxdy should use)
    grid_phys: torch.Tensor


    # optional: true velocity for supervision (B,2,H,W)
    u_true: Optional[torch.Tensor] = None
class PatchTimeWindowIterable(IterableDataset):
    """
    Randomly sample (case, start_time, patch_index) -> one batch.
    IterableDataset avoids building huge indices in memory.
    """

    def __init__(
        self,
        case_dirs: List[Path],
        *,
        T_in: int,
        T_out: int,
        batch_size: int,
        steps_per_epoch: int,
        seed: int = 0,
        cache_size: int = 8,
    ):
        super().__init__()
        self.case_dirs = [Path(p) for p in case_dirs]
        self.T_in = int(T_in)
        self.T_out = int(T_out)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        self.cache_size = int(cache_size)

        self._epoch: int = 0

        if self.T_in <= 0:
            raise ValueError("T_in must be > 0")
        if self.T_out <= 0:
            raise ValueError("T_out must be > 0")

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[Batch]:
        wi = get_worker_info()
        worker_id = int(wi.id) if wi is not None else 0
        worker_count = int(wi.num_workers) if wi is not None else 1
        # torch provides a per-worker initial_seed; mix it in so workers don't sample identical windows
        torch_seed = int(torch.initial_seed()) if wi is not None else 0
        seed = (self.seed + 12345 + 1000003 * self._epoch + 10007 * worker_id + torch_seed) % (2**32)
        rng = np.random.default_rng(seed)
        cache = NPZCache(max_items=self.cache_size)

        # pre-list files per case
        case_files: List[List[Path]] = []
        for d in self.case_dirs:
            files = sorted(d.glob("patch_t*.npz"))
            if files:
                case_files.append(files)
        if not case_files:
            raise RuntimeError("No patch_t*.npz found in given case_dirs")

        worker_steps = (self.steps_per_epoch + worker_count - 1 - worker_id) // worker_count
        for _ in range(worker_steps):
            ci = int(rng.integers(0, len(case_files)))
            files = case_files[ci]
            n_files = len(files)

            # need indices up to s + T_in + T_out - 2 for X, and up to s + T_in + T_out - 2 for Y
            # so total required files: T_in + T_out - 1
            max_start = n_files - (self.T_in + self.T_out - 1)
            if max_start < 0:
                raise RuntimeError(
                    f"case={self.case_dirs[ci].name} not enough steps: n_files={n_files} < T_in+T_out-1={self.T_in + self.T_out - 1}"
                )
            s = int(rng.integers(0, max_start + 1))

            X0, _, meta0 = cache.get(files[s])
            Np, C, H, W = X0.shape
            if C < 4:
                raise RuntimeError(f"{files[s]}: X has too few channels: C={C}")
            n_phys = int(C - 3)

            pidx = rng.choice(Np, size=self.batch_size, replace=(self.batch_size > Np))

            # grid from X(s): normalized coords in [0,1]
            grid_norm = X0[pidx][:, -3:, :, :].astype(np.float32)  # [x_norm,y_norm,mask]

            # Build a **physical** copy for PDE/gh usage using per-case metadata (recommended for multi-case).
            # We keep the model input grid as normalized (grid_norm) so the network always sees a consistent range.
            grid_phys = grid_norm.copy()
            mi: Dict[str, Any] = {}
            if isinstance(meta0, dict):
                mi = meta0.get("meta_in", meta0)
            try:
                x0 = float(mi.get("x0", 0.0))
                y0 = float(mi.get("y0", 0.0))
                Lx = float(mi.get("Lx", 1.0))
                Ly = float(mi.get("Ly", 1.0))
                grid_phys[:, 0:1, :, :] = x0 + grid_norm[:, 0:1, :, :] * Lx
                grid_phys[:, 1:2, :, :] = y0 + grid_norm[:, 1:2, :, :] * Ly
            except Exception:
                # If meta is missing/malformed, fall back to using normalized coords as "physical".
                # (It will still be self-consistent, just with wrong units.)
                pass



            # history frames: X(s .. s+T_in-1) -> phys_hist
            phys_hist = np.empty((self.batch_size, self.T_in, n_phys, H, W), dtype=np.float32)
            for ti in range(self.T_in):
                Xi, _, _ = cache.get(files[s + ti])
                phys_hist[:, ti, :, :, :] = Xi[pidx][:, :n_phys, :, :]

            # future truth frames: X(s+T_in .. s+T_in+T_out-2) -> phys_fut (length T_out-1)
            if self.T_out > 1:
                phys_fut = np.empty((self.batch_size, self.T_out - 1, n_phys, H, W), dtype=np.float32)
                for k in range(self.T_out - 1):
                    Xk, _, _ = cache.get(files[s + self.T_in + k])
                    phys_fut[:, k, :, :, :] = Xk[pidx][:, :n_phys, :, :]
            else:
                phys_fut = np.empty((self.batch_size, 0, n_phys, H, W), dtype=np.float32)

            # targets: Y(s+T_in-1 .. s+T_in+T_out-2) -> dp_fut
            dp_fut = np.empty((self.batch_size, self.T_out, 1, H, W), dtype=np.float32)
            for k in range(self.T_out):
                _, Yk, _ = cache.get(files[s + self.T_in - 1 + k])
                dp_fut[:, k, :, :, :] = Yk[pidx][:, 0:1, :, :]

            yield Batch(
                phys_hist=torch.from_numpy(phys_hist),
                phys_fut=torch.from_numpy(phys_fut),
                dp_fut=torch.from_numpy(dp_fut),
                grid=torch.from_numpy(grid_norm),
                grid_phys=torch.from_numpy(grid_phys),
            )

class PatchCorrectorWindowIterable(IterableDataset):
    """
    Sample (case, physical_time_file, spatial_patch) and build a k-window over pressure correctors.

    patch_tXXXX.npz (from build_pressure_corrector_dataset.py) contains:
      X: (N, C, H, W) with channels [..., x_norm, y_norm, mask]
      Y: (N, 1, H, W) = Δp_rgh for that corrector step
      coords: (N, 4) = [time_index, k, y0, x0]

    Output Batch:
      phys_hist: (B, T_in, n_phys, H, W)  frames at k = s .. s+T_in-1
      phys_fut:  (B, T_out-1, n_phys, H, W) frames at k = s+T_in .. s+T_in+T_out-2
      dp_fut:    (B, T_out, 1, H, W)  Δp_rgh at k = s+T_in-1 .. s+T_in+T_out-2

    Notes:
      - We set s=0 by default to match online deployment (k starts from 0 each time step).
      - If u_patch_root is provided, the loader expects aligned files:
          u_patch_root/<case>/patch_tXXXX.npz containing array 'U' with shape (N,2,H,W).
        It will return u_true = U at the *final* corrector in the window.
    """

    def __init__(
        self,
        case_dirs: List[Path],
        *,
        T_in: int,
        T_out: int,
        batch_size: int,
        steps_per_epoch: int,
        seed: int = 0,
        cache_size: int = 8,
        u_patch_root: Optional[Path] = None,
    ):
        super().__init__()
        self.case_dirs = [Path(p) for p in case_dirs]
        self.T_in = int(T_in)
        self.T_out = int(T_out)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        self.cache_size = int(cache_size)
        self.u_patch_root = Path(u_patch_root) if u_patch_root is not None else None

        self._epoch: int = 0

        if self.T_in <= 0:
            raise ValueError("T_in must be > 0")
        if self.T_out <= 0:
            raise ValueError("T_out must be > 0")

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[Batch]:
        wi = get_worker_info()
        worker_id = int(wi.id) if wi is not None else 0
        worker_count = int(wi.num_workers) if wi is not None else 1
        torch_seed = int(torch.initial_seed()) if wi is not None else 0
        seed = (self.seed + 12345 + 1000003 * self._epoch + 10007 * worker_id + torch_seed) % (2**32)
        rng = np.random.default_rng(seed)
        cache = NPZCache(max_items=self.cache_size)

        # pre-list files per case
        case_files: List[List[Path]] = []
        for d in self.case_dirs:
            files = sorted(d.glob("patch_t*.npz"))
            if files:
                case_files.append(files)
        if not case_files:
            raise RuntimeError("No patch_t*.npz found in given case_dirs")

        worker_steps = (self.steps_per_epoch + worker_count - 1 - worker_id) // worker_count
        for _ in range(worker_steps):
            ci = int(rng.integers(0, len(case_files)))
            files = case_files[ci]
            fi = int(rng.integers(0, len(files)))
            f = files[fi]

            X0, Y0, coords0, meta0 = cache.get_full(f)
            if coords0 is None:
                raise RuntimeError(f"{f}: coords missing. Need coords=[time,k,y0,x0].")

            N, C, H, W = X0.shape
            if C < 4:
                raise RuntimeError(f"{f}: X has too few channels: C={C}")
            n_phys = int(C - 3)

            # group by spatial patch (y0,x0)
            yx = coords0[:, 2:4].astype(np.int32)   # (N,2)
            uniq = np.unique(yx, axis=0)
            if uniq.shape[0] < 1:
                raise RuntimeError(f"{f}: no patch groups")

            gidx = rng.choice(uniq.shape[0], size=self.batch_size, replace=(self.batch_size > uniq.shape[0]))

            phys_hist = np.empty((self.batch_size, self.T_in, n_phys, H, W), dtype=np.float32)
            phys_fut = np.empty((self.batch_size, max(self.T_out - 1, 0), n_phys, H, W), dtype=np.float32)
            dp_fut = np.empty((self.batch_size, self.T_out, 1, H, W), dtype=np.float32)

            grid_norm = np.empty((self.batch_size, 3, H, W), dtype=np.float32)
            grid_phys = np.empty((self.batch_size, 3, H, W), dtype=np.float32)

            u_true = None
            Uall = None
            u_row_of_dp = None  # optional: map dp-row -> u-row when U patches keep only a subset (e.g. final k)
            if self.u_patch_root is not None:
                ufile = (self.u_patch_root / self.case_dirs[ci].name / f.name)
                with np.load(ufile, allow_pickle=False) as z:
                    if "U" not in z.files:
                        raise RuntimeError(f"{ufile}: expected array 'U' (Nu,2,H,W)")
                    Uall = z["U"]
                    dp_index = z["dp_index"].astype(np.int64) if ("dp_index" in z.files) else None

                if Uall.ndim != 4 or Uall.shape[1] != 2:
                    raise RuntimeError(f"{ufile}: bad U shape {getattr(Uall, 'shape', None)}; expected (Nu,2,H,W)")

                if Uall.shape[0] == N:
                    u_row_of_dp = None  # fully aligned: dp-row == u-row
                else:
                    # Allow "final-k only" U patches: file stores only a subset of dp rows.
                    # In that case, it must also provide dp_index, listing which dp rows are kept.
                    if dp_index is None:
                        raise RuntimeError(
                            f"{ufile}: U N={Uall.shape[0]} does not match X N={N}. "
                            "If you built U patches with k_mode='final', make sure to save dp_index in the .npz."
                        )
                    if dp_index.shape[0] != Uall.shape[0]:
                        raise RuntimeError(
                            f"{ufile}: dp_index N={dp_index.shape[0]} does not match U N={Uall.shape[0]}"
                        )
                    if dp_index.size == 0:
                        raise RuntimeError(f"{ufile}: dp_index is empty")
                    if dp_index.min() < 0 or dp_index.max() >= N:
                        raise RuntimeError(
                            f"{ufile}: dp_index out of range [0,{N-1}] (min={dp_index.min()}, max={dp_index.max()})"
                        )
                    u_row_of_dp = -np.ones((N,), dtype=np.int64)
                    u_row_of_dp[dp_index] = np.arange(Uall.shape[0], dtype=np.int64)

                u_true = np.empty((self.batch_size, 2, H, W), dtype=np.float32)

            # meta for physical coords, consistent with PatchTimeWindowIterable
            mi: Dict[str, Any] = meta0.get("meta_in", meta0) if isinstance(meta0, dict) else {}
            x0m = float(mi.get("x0", 0.0))
            y0m = float(mi.get("y0", 0.0))
            Lxm = float(mi.get("Lx", 1.0))
            Lym = float(mi.get("Ly", 1.0))

            for bi, gi in enumerate(gidx):
                y0i, x0i = uniq[gi]
                rows = np.where((yx[:, 0] == y0i) & (yx[:, 1] == x0i))[0]
                ks = coords0[rows, 1].astype(np.int32)

                Kmax = int(ks.max()) + 1
                k_to_row = -np.ones((Kmax,), dtype=np.int64)
                for r, k in zip(rows, ks):
                    if 0 <= k < Kmax:
                        k_to_row[int(k)] = int(r)

                need = self.T_in + self.T_out - 1
                if Kmax < need or np.any(k_to_row[:need] < 0):
                    raise RuntimeError(
                        f"{f}: patch(y0={y0i},x0={x0i}) has incomplete k sequence. "
                        f"Kmax={Kmax}, need={need}."
                    )

                s = 0  # fixed start (k=0)

                rid0 = int(k_to_row[s])
                g = X0[rid0, -3:, :, :].astype(np.float32)  # [x_norm,y_norm,mask]
                grid_norm[bi] = g

                # physical coords in meters
                grid_phys[bi] = g
                grid_phys[bi, 0:1] = x0m + g[0:1] * Lxm
                grid_phys[bi, 1:2] = y0m + g[1:2] * Lym

                for ti in range(self.T_in):
                    rid = int(k_to_row[s + ti])
                    phys_hist[bi, ti] = X0[rid, :n_phys].astype(np.float32)

                if self.T_out > 1:
                    for kk in range(self.T_out - 1):
                        rid = int(k_to_row[s + self.T_in + kk])
                        phys_fut[bi, kk] = X0[rid, :n_phys].astype(np.float32)

                for kk in range(self.T_out):
                    rid = int(k_to_row[s + self.T_in - 1 + kk])
                    dp_fut[bi, kk] = Y0[rid, 0:1].astype(np.float32)

                if u_true is not None:
                    k_final = s + self.T_in + self.T_out - 2
                    rid_u = int(k_to_row[k_final])
                    if u_row_of_dp is None:
                        u_true[bi] = Uall[rid_u].astype(np.float32)
                    else:
                        urow = int(u_row_of_dp[rid_u])
                        if urow < 0:
                            raise RuntimeError(f"{ufile}: missing U row for dp_row={rid_u} (did you build U patches with k_mode='final'?)")
                        u_true[bi] = Uall[urow].astype(np.float32)

            yield Batch(
                phys_hist=torch.from_numpy(phys_hist),
                phys_fut=torch.from_numpy(phys_fut),
                dp_fut=torch.from_numpy(dp_fut),
                grid=torch.from_numpy(grid_norm),
                grid_phys=torch.from_numpy(grid_phys),
                u_true=torch.from_numpy(u_true) if u_true is not None else None,
            )
