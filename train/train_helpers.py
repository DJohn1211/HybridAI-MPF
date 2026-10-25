"""
Small, focused helpers for the dp_rgh / Δp_rgh time-window trainer.

This module intentionally contains *only* reusable, side-effect-free utilities:
- channel bookkeeping
- p/p_rgh consistency recompute (via rho*gh)
- normalization helpers for window inputs / outputs
- gauge (constant-mode) utilities
- schedules (teacher forcing, curriculum, PDE/cont ramps)
- offset augmentation
- LR schedule + checkpoint I/O

The goal is to keep `trainer_timewindow_dp_rgh.py` readable: training logic lives there,
mechanics live here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, TYPE_CHECKING

import math
import torch
import torch.nn.functional as F

from data_timewindow import Batch
from losses import inner_mask, get_rho_from_alpha
from utils import capture_rng_state, restore_rng_state

if TYPE_CHECKING:
    from trainer_timewindow_dp_rgh import TrainConfig


def _attach_learned_logvars(model: torch.nn.Module, cfg: TrainConfig, device: torch.device) -> None:
    """Attach learned log-variance parameters to the model (saved in state_dict).

    dp remains the main supervised objective; logvars only reweight auxiliary terms (pde/cont).
    """
    tasks = tuple(getattr(cfg, "learned_logvar_tasks", ("pde", "cont")))
    if not tasks:
        return
    if hasattr(model, "loss_logvars") and isinstance(getattr(model, "loss_logvars"), torch.nn.ParameterDict):
        return
    init = float(getattr(cfg, "learned_logvar_init", 0.0))
    pd = torch.nn.ParameterDict()
    for t in tasks:
        pd[str(t)] = torch.nn.Parameter(torch.tensor(init, device=device, dtype=torch.float32))
    model.loss_logvars = pd  # type: ignore[attr-defined]

def _clamp_learned_logvars(model: torch.nn.Module, cfg: TrainConfig) -> None:
    """Clamp log-variance parameters to a safe range to avoid runaway weights."""
    if not (hasattr(model, "loss_logvars") and isinstance(getattr(model, "loss_logvars"), torch.nn.ParameterDict)):
        return
    lo = float(getattr(cfg, "learned_logvar_min", -8.0))
    hi = float(getattr(cfg, "learned_logvar_max", 8.0))
    for p in model.loss_logvars.values():  # type: ignore[attr-defined]
        p.data.clamp_(lo, hi)

def _logvar_weighted_term(base_weighted_loss: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Return exp(-s)*L + s (uncertainty weighting)."""
    return torch.exp(-s) * base_weighted_loss + s


# =========================
# Config
# =========================

def _resolve_channel(idx: int, n_phys: int) -> int:
    """Allow negative indices for physical channel selection."""
    if idx < 0:
        idx = n_phys + idx
    if not (0 <= idx < n_phys):
        raise ValueError(f"Channel index out of range: idx={idx}, n_phys={n_phys}")
    return int(idx)

def _compute_gh(cfg: "TrainConfig", grid: torch.Tensor) -> torch.Tensor:
    """Compute gh = g·C - ghRef.

    grid: (B,3,H,W) with channels [x, y, mask].
      - If cfg.grid_is_normalized=True: x,y are normalized in [0,1] and we convert using domain_{x0,y0,Lx,Ly}.
      - If cfg.grid_is_normalized=False: x,y are already **physical coordinates** (recommended; supports per-case domains).
    """
    x = grid[:, 0:1].float()
    y = grid[:, 1:2].float()

    if bool(getattr(cfg, "grid_is_normalized", False)):
        x = float(cfg.domain_x0) + x * float(cfg.domain_Lx)
        y = float(cfg.domain_y0) + y * float(cfg.domain_Ly)

    return float(cfg.g_x) * x + float(cfg.g_y) * y - float(cfg.gh_ref)

def _rho_from_frame(
        cfg: "TrainConfig",
        frame: torch.Tensor,  # (B,n_phys,H,W)
        *,
        alpha_ch: int,
        rho_ch: Optional[int],
        invrho_ch: Optional[int],
) -> torch.Tensor:
    """Get rho_mix from a physical frame."""
    if rho_ch is not None:
        return frame[:, rho_ch:rho_ch + 1].float()
    if invrho_ch is not None:
        invrho = frame[:, invrho_ch:invrho_ch + 1].float()
        return 1.0 / (invrho + 1e-6)
    alpha = frame[:, alpha_ch:alpha_ch + 1].float()
    return get_rho_from_alpha(alpha, rho1=float(cfg.rho1), rho2=float(cfg.rho2))

def _overwrite_prgh_if_needed(
        cfg: "TrainConfig",
        *,
        next_frame: torch.Tensor,  # (B,n_phys,H,W)
        p_for_prgh: torch.Tensor,  # (B,1,H,W)  the p you will feed into the next window (TF/pred)
        grid: torch.Tensor,  # (B,3,H,W)
        alpha_ch: int,
        rho_ch: Optional[int],
        invrho_ch: Optional[int],
        prgh_ch: Optional[int],
) -> None:
    """In-place overwrite of p_rgh channel to enforce p_rgh = p - rho*gh."""
    if prgh_ch is None:
        return
    if not bool(getattr(cfg, "recompute_prgh_in_rollout", True)):
        return
    rho = _rho_from_frame(cfg, next_frame, alpha_ch=alpha_ch, rho_ch=rho_ch, invrho_ch=invrho_ch)
    gh = _compute_gh(cfg, grid)
    next_frame[:, prgh_ch:prgh_ch + 1] = p_for_prgh - rho * gh

def _overwrite_p_if_needed(
        cfg: "TrainConfig",
        *,
        next_frame: torch.Tensor,  # (B,n_phys,H,W)
        prgh_for_p: torch.Tensor,  # (B,1,H,W) the p_rgh you will feed into the next window (TF/pred)
        grid: torch.Tensor,  # (B,3,H,W)
        alpha_ch: int,
        rho_ch: Optional[int],
        invrho_ch: Optional[int],
        p_ch: Optional[int],
) -> None:
    """In-place overwrite of p channel to enforce p = p_rgh + rho*gh."""
    if p_ch is None:
        return
    if not bool(getattr(cfg, "recompute_p_in_rollout", True)):
        return
    rho = _rho_from_frame(cfg, next_frame, alpha_ch=alpha_ch, rho_ch=rho_ch, invrho_ch=invrho_ch)
    gh = _compute_gh(cfg, grid)
    next_frame[:, p_ch:p_ch + 1] = prgh_for_p + rho * gh

def _repeat_norm(mean: torch.Tensor, std: torch.Tensor, T_in: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    mean/std: (n_phys,) -> (1, n_phys*T_in, 1, 1)
    """
    n_phys = int(mean.numel())
    mean_rep = mean.view(1, n_phys, 1, 1).repeat(1, T_in, 1, 1).view(1, n_phys * T_in, 1, 1)
    std_rep = std.view(1, n_phys, 1, 1).repeat(1, T_in, 1, 1).view(1, n_phys * T_in, 1, 1)
    return mean_rep, std_rep

def build_x_from_window(
        phys_win: torch.Tensor,  # (B,T_in,n_phys,H,W)
        grid: torch.Tensor,  # (B,3,H,W) [x,y,mask]
        *,
        x_mean: torch.Tensor,  # (n_phys,)
        x_std: torch.Tensor,  # (n_phys,)
        T_in: int,
        normalize_channels: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Return model input with only dynamic channels normalized.

    The paper keeps static geometry channels in their original representation.
    ``normalize_channels=None`` preserves the legacy behavior of normalizing all
    physical channels; the paper configuration supplies channels 0..4 and leaves
    ``invDx2``/``invDy2`` unchanged.
    """
    B, Tin, n_phys, H, W = phys_win.shape
    if Tin != T_in:
        raise ValueError(f"phys_win Tin expected {T_in}, got {Tin}")
    if x_mean.numel() != n_phys or x_std.numel() != n_phys:
        raise ValueError(f"NormStats mismatch: mean/std length {x_mean.numel()} vs n_phys {n_phys}")

    phys = phys_win.reshape(B, Tin * n_phys, H, W)
    mean, std = _repeat_norm(x_mean, x_std, T_in)
    if normalize_channels is None:
        phys_n = (phys - mean) / (std + 1e-12)
    else:
        selected = {int(c) for c in normalize_channels}
        invalid = sorted(c for c in selected if c < 0 or c >= n_phys)
        if invalid:
            raise ValueError(
                f"normalize_channels contains invalid indices {invalid}; n_phys={n_phys}"
            )
        channel_mask = torch.tensor(
            [1.0 if (c % n_phys) in selected else 0.0 for c in range(Tin * n_phys)],
            device=phys.device,
            dtype=phys.dtype,
        ).view(1, Tin * n_phys, 1, 1)
        normalized = (phys - mean) / (std + 1e-12)
        phys_n = normalized * channel_mask + phys * (1.0 - channel_mask)
    return torch.cat([phys_n, grid], dim=1)

def y_phys_to_norm(y_phys: torch.Tensor, *, y_mean: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
    y_phys = y_phys.float()
    y_mean = y_mean.float()
    y_std = y_std.float()
    return (y_phys - y_mean) / (y_std + 1e-12)

def y_norm_to_phys(y_norm: torch.Tensor, *, y_mean: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
    y_norm = y_norm.float()
    y_mean = y_mean.float()
    y_std = y_std.float()
    return y_norm * (y_std + 1e-12) + y_mean

# =========================
# Helpers: gauge / masks
# =========================

def _masked_mean2d(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-sample masked mean over (H,W). x/mask: (B,1,H,W)."""
    x = x.float()
    m = mask.float()
    denom = m.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    return (x * m).sum(dim=(2, 3), keepdim=True) / denom

def _demean2d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Subtract per-sample masked mean (removes constant/gauge mode)."""
    return x.float() - _masked_mean2d(x, mask)

def _match_mean2d(x: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Shift x by a constant so its masked mean matches ref's masked mean."""
    x = x.float()
    ref = ref.float()
    return x + (_masked_mean2d(ref, mask) - _masked_mean2d(x, mask))

def _get_gauge_masks(cfg: TrainConfig, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (mean_mask, loss_mask). mean_mask is always an inner mask."""
    base_margin = getattr(cfg, "gauge_interior_margin", None)
    if base_margin is None:
        base_margin = int(getattr(cfg, "poisson_check_interior_margin", 1))
    mean_mask = inner_mask(mask.float(), margin=int(base_margin))
    if bool(getattr(cfg, "gauge_use_interior_mask", True)):
        loss_mask = mean_mask
    else:
        loss_mask = mask.float()
    return mean_mask, loss_mask

def _infer_dx_dy_from_invDx2(invDx2: torch.Tensor, invDy2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (dx,dy) as (B,1,1,1) from invDx2/invDy2 fields."""
    # invDx2 ~ 1/dx^2; clamp to avoid inf/NaN
    dx = (invDx2.mean(dim=(2, 3), keepdim=True).clamp(1e-30, 1e12)).rsqrt()
    dy = (invDy2.mean(dim=(2, 3), keepdim=True).clamp(1e-30, 1e12)).rsqrt()
    return dx, dy

def _grad2d(p: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Central-diff gradient for p (B,1,H,W) with scalar dx/dy broadcastable."""
    if p.ndim != 4 or p.size(1) != 1:
        raise ValueError(f"p must be (B,1,H,W), got {tuple(p.shape)}")
    B, _, H, W = p.shape
    dpdx = torch.zeros_like(p)
    dpdy = torch.zeros_like(p)
    # x (W)
    if W >= 3:
        dpdx[:, :, :, 1:-1] = (p[:, :, :, 2:] - p[:, :, :, :-2]) / (2.0 * dx)
        dpdx[:, :, :, 0:1] = (p[:, :, :, 1:2] - p[:, :, :, 0:1]) / (dx)
        dpdx[:, :, :, -1:] = (p[:, :, :, -1:] - p[:, :, :, -2:-1]) / (dx)
    elif W == 2:
        dpdx[:, :, :, 0:1] = (p[:, :, :, 1:2] - p[:, :, :, 0:1]) / (dx)
        dpdx[:, :, :, 1:2] = dpdx[:, :, :, 0:1]
    # y (H)
    if H >= 3:
        dpdy[:, :, 1:-1, :] = (p[:, :, 2:, :] - p[:, :, :-2, :]) / (2.0 * dy)
        dpdy[:, :, 0:1, :] = (p[:, :, 1:2, :] - p[:, :, 0:1, :]) / (dy)
        dpdy[:, :, -1:, :] = (p[:, :, -1:, :] - p[:, :, -2:-1, :]) / (dy)
    elif H == 2:
        dpdy[:, :, 0:1, :] = (p[:, :, 1:2, :] - p[:, :, 0:1, :]) / (dy)
        dpdy[:, :, 1:2, :] = dpdy[:, :, 0:1, :]
    return dpdx, dpdy

def _pde_weight(cfg: TrainConfig, epoch: int) -> float:
    if cfg.pde_loss_weight <= 0:
        return 0.0
    if epoch < cfg.pde_warmup_epochs:
        return 0.0
    if cfg.pde_ramp_epochs <= 0:
        return float(cfg.pde_loss_weight)
    k = min(1.0, max(0.0, (epoch - cfg.pde_warmup_epochs) / float(cfg.pde_ramp_epochs)))
    return float(cfg.pde_loss_weight) * k

def _cont_weight(cfg: TrainConfig, epoch: int) -> float:
    """Warmup + linear ramp for continuity loss."""
    if float(getattr(cfg, "cont_loss_weight", 0.0)) <= 0:
        return 0.0
    warm = int(getattr(cfg, "cont_warmup_epochs", 0))
    ramp = int(getattr(cfg, "cont_ramp_epochs", 0))
    if epoch < warm:
        return 0.0
    if ramp <= 0:
        return float(getattr(cfg, "cont_loss_weight", 0.0))
    k = min(1.0, max(0.0, (epoch - warm) / float(ramp)))
    return float(getattr(cfg, "cont_loss_weight", 0.0)) * k

def _div2d(Ux: torch.Tensor, Uy: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """Central-diff divergence for (Ux,Uy) each (B,1,H,W)."""
    Ux = Ux.float()
    Uy = Uy.float()
    if Ux.ndim != 4 or Uy.ndim != 4 or Ux.size(1) != 1 or Uy.size(1) != 1:
        raise ValueError(f"Ux,Uy must be (B,1,H,W), got {tuple(Ux.shape)} and {tuple(Uy.shape)}")
    dUxdx = torch.zeros_like(Ux)
    dUydy = torch.zeros_like(Uy)
    _, _, H, W = Ux.shape
    if W >= 3:
        dUxdx[:, :, :, 1:-1] = (Ux[:, :, :, 2:] - Ux[:, :, :, :-2]) / (2.0 * dx)
    if H >= 3:
        dUydy[:, :, 1:-1, :] = (Uy[:, :, 2:, :] - Uy[:, :, :-2, :]) / (2.0 * dy)
    return dUxdx + dUydy

def _teacher_forcing_ratio(cfg: TrainConfig, epoch: int) -> float:
    decay_epochs = max(1, int(round(cfg.epochs * cfg.tf_decay_portion)))
    if epoch <= 0:
        tf_ratio = cfg.tf_start
    elif epoch >= decay_epochs:
        tf_ratio = cfg.tf_end
    else:
        k = epoch / float(decay_epochs)
        tf_ratio = cfg.tf_start + (cfg.tf_end - cfg.tf_start) * k
    return float(max(0.0, min(1.0, tf_ratio)))

def _teacher_forcing_ratio_step(cfg: TrainConfig, epoch: int, t: int) -> float:
    """Step-wise TF: later rollout steps get less TF to better match long-horizon rollout."""
    tf_epoch = _teacher_forcing_ratio(cfg, epoch)
    gamma = float(getattr(cfg, 'tf_step_gamma', 1.0))
    tf_end_step = float(getattr(cfg, 'tf_step_end', cfg.tf_end))
    tf_t = tf_end_step + (tf_epoch - tf_end_step) * (gamma ** int(t))
    return float(max(0.0, min(1.0, tf_t)))

def _tout_for_epoch(cfg: TrainConfig, epoch: int) -> int:
    """Curriculum on rollout horizon."""
    tout = int(cfg.T_out)
    if not bool(getattr(cfg, 'use_curriculum', False)):
        return max(1, tout)

    e1 = int(getattr(cfg, 'curriculum_epoch1', 0))
    e2 = int(getattr(cfg, 'curriculum_epoch2', 0))
    t1 = int(getattr(cfg, 'curriculum_tout1', 1))
    t2 = int(getattr(cfg, 'curriculum_tout2', tout))

    if epoch < e1:
        tout = min(tout, t1)
    elif epoch < e2:
        tout = min(tout, t2)
    else:
        tout = tout

    return max(1, int(tout))

def _shift_crop_4d(x: torch.Tensor, *, di: int, dj: int, pad: int, mode: str, value: float = 0.0) -> torch.Tensor:
    """Shift by (di,dj) via pad+crop. x: (B,C,H,W)."""
    B, C, H, W = x.shape
    if pad <= 0 or (di == 0 and dj == 0):
        return x
    if mode == 'constant':
        xp = F.pad(x, (pad, pad, pad, pad), mode=mode, value=float(value))
    else:
        xp = F.pad(x, (pad, pad, pad, pad), mode=mode)
    si = pad + int(di)
    sj = pad + int(dj)
    return xp[:, :, si:si + H, sj:sj + W]

def _shift_crop_5d(x: torch.Tensor, *, di: int, dj: int, pad: int, mode: str, value: float = 0.0) -> torch.Tensor:
    """Shift by (di,dj) for x: (B,T,C,H,W) via flatten+crop."""
    B, T, C, H, W = x.shape
    y = x.reshape(B * T, C, H, W)
    y = _shift_crop_4d(y, di=di, dj=dj, pad=pad, mode=mode, value=value)
    return y.reshape(B, T, C, H, W)

def _maybe_offset_augment(batch: Batch, *, max_shift: int, prob: float) -> Batch:
    """Random integer offset crop to reduce patch-boundary artifacts.

    - Uses replicate padding for physical fields and x/y.
    - Uses constant-0 padding for mask so cropped-in padded regions become invalid.

    This augmentation effectively makes the model see the same physics at slightly different patch alignments,
    which usually reduces blocky rollout/stitch artifacts.
    """
    max_shift = int(max_shift)
    prob = float(prob)
    if max_shift <= 0 or prob <= 0:
        return batch

    if float(torch.rand((), device=batch.phys_hist.device)) >= prob:
        return batch

    di = int(torch.randint(-max_shift, max_shift + 1, (1,), device=batch.phys_hist.device).item())
    dj = int(torch.randint(-max_shift, max_shift + 1, (1,), device=batch.phys_hist.device).item())
    if di == 0 and dj == 0:
        return batch

    pad = max_shift

    phys_hist = _shift_crop_5d(batch.phys_hist, di=di, dj=dj, pad=pad, mode='replicate')
    phys_fut = _shift_crop_5d(batch.phys_fut, di=di, dj=dj, pad=pad, mode='replicate')
    dp_fut = _shift_crop_5d(batch.dp_fut, di=di, dj=dj, pad=pad, mode='replicate')

    # grid_norm and grid_phys: [x,y,mask]
    xy = _shift_crop_4d(batch.grid[:, 0:2], di=di, dj=dj, pad=pad, mode='replicate')
    m = _shift_crop_4d(batch.grid[:, 2:3], di=di, dj=dj, pad=pad, mode='constant', value=0.0)
    grid = torch.cat([xy, m], dim=1)

    xy_p = _shift_crop_4d(batch.grid_phys[:, 0:2], di=di, dj=dj, pad=pad, mode='replicate')
    m_p = _shift_crop_4d(batch.grid_phys[:, 2:3], di=di, dj=dj, pad=pad, mode='constant', value=0.0)
    grid_phys = torch.cat([xy_p, m_p], dim=1)

    u_true = None
    if getattr(batch, "u_true", None) is not None:
        # U is (B,2,H,W). Shift/crop it the same way so continuity & U-supervision stay consistent under aug.
        u_true = _shift_crop_4d(batch.u_true, di=di, dj=dj, pad=pad, mode="replicate")

    return Batch(
        phys_hist=phys_hist,
        phys_fut=phys_fut,
        dp_fut=dp_fut,
        grid=grid,
        grid_phys=grid_phys,
        u_true=u_true,
    )

def _set_optimizer_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    lr = float(lr)
    for pg in opt.param_groups:
        pg['lr'] = lr

def _warmup_cosine_lr(*, base_lr: float, min_lr: float, step: int, total_steps: int, warmup_steps: int) -> float:
    """Linear warmup then cosine decay."""
    base_lr = float(base_lr)
    min_lr = float(min_lr)
    step = int(step)
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)

    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)

    # after warmup
    if total_steps <= warmup_steps:
        return min_lr
    t = min(max(step - warmup_steps, 0), total_steps - warmup_steps)
    frac = float(t) / float(total_steps - warmup_steps)
    # cosine from base_lr -> min_lr
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * frac))

def _lr_for_step(cfg: TrainConfig, step: int, total_steps: int, warmup_steps: int) -> float:
    if getattr(cfg, 'lr_schedule', 'constant') == 'warmup_cosine':
        return _warmup_cosine_lr(base_lr=cfg.lr, min_lr=getattr(cfg, 'min_lr', 0.0), step=step, total_steps=total_steps,
                                 warmup_steps=warmup_steps)
    return float(cfg.lr)

def _optimizer_to_device(opt: torch.optim.Optimizer, device: torch.device) -> None:
    """After opt.load_state_dict, AdamW states are on CPU; move them to the model device."""
    for st in opt.state.values():
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                st[k] = v.to(device=device)

def _save_checkpoint(path: Path, *, model: torch.nn.Module, opt: torch.optim.Optimizer,
                     scaler: Optional[torch.cuda.amp.GradScaler],
                     epoch: int, global_step: int, best_val: float, best_rel: Optional[float] = None, cfg: TrainConfig) -> None:
    ckpt = {
        'epoch': int(epoch),
        'global_step': int(global_step),
        'best_val': float(best_val),
        'best_rel': (float(best_rel) if best_rel is not None else None),
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'scaler': (scaler.state_dict() if scaler is not None else None),
        'cfg': {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()},
        'rng_state': capture_rng_state(),
    }
    torch.save(ckpt, path)

def _export_libtorch_state_dict(path: Path, *, model: torch.nn.Module) -> None:
    """Save a plain CPU state_dict for C++ libtorch `torch::load(net, path)`.

    We move tensors to CPU before saving, because C++ `torch::load` does not have a
    Python-like `map_location` argument, and CPU weights are the most portable.

    NOTE: Drop training-only keys (e.g. learned logvars) to keep C++ loading strict-safe.
    """
    sd = model.state_dict()
    sd = {k: v for k, v in sd.items() if not str(k).startswith("loss_logvars.")}
    sd_cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in sd.items()}
    torch.save(sd_cpu, path)

def _try_load_checkpoint(path: Path, *, model: torch.nn.Module, opt: torch.optim.Optimizer,
                         scaler: Optional[torch.cuda.amp.GradScaler], device: torch.device,
                         expected_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model' in ckpt:
        if expected_cfg is not None and isinstance(ckpt.get('cfg'), dict):
            saved_cfg = ckpt['cfg']
            mismatches = {
                key: (saved_cfg.get(key), expected)
                for key, expected in expected_cfg.items()
                if saved_cfg.get(key) != expected
            }
            if mismatches:
                details = ", ".join(
                    f"{key}: checkpoint={old!r}, current={new!r}"
                    for key, (old, new) in sorted(mismatches.items())
                )
                raise ValueError(
                    "Refusing to resume a checkpoint with incompatible paper settings: "
                    + details
                )
        # tolerant to added/removed keys (e.g., learned logvars)
        model.load_state_dict(ckpt['model'], strict=False)

        if 'opt' in ckpt and ckpt['opt'] is not None:
            try:
                opt.load_state_dict(ckpt['opt'])
                _optimizer_to_device(opt, device)
            except Exception as e:
                print(f"[warn] optimizer state not loaded (param set changed): {e}")

        if scaler is not None and ckpt.get('scaler') is not None:
            try:
                scaler.load_state_dict(ckpt['scaler'])
            except Exception as e:
                print(f"[warn] scaler state not loaded: {e}")
        if isinstance(ckpt.get('rng_state'), dict):
            restore_rng_state(ckpt['rng_state'])
        return ckpt
    # fallback: allow loading plain state_dict checkpoint as model-only
    model.load_state_dict(ckpt, strict=False)
    return {'epoch': -1, 'global_step': 0, 'best_val': float('inf')}
