"""Train a pressure-corrector FNO-ResUNet with scheduled rollout.

The paper configuration uses corrector-index windows within one physical time.
Inputs are [alpha, rhs, rAU, p_in, p_rgh_in, invDx2, invDy2, x, y, mask],
and targets are the corrector increments delta_p_rgh. The final three input
channels contain normalized x/y coordinates and the validity mask.

The model predicts normalized pressure increments. The trainer reconstructs
the pressure state, applies scheduled teacher forcing, and recomputes the
paired pressure channel from the hydrostatic contribution during rollout.
Supervised, gradient, reconstruction, spectral, Poisson, and continuity terms
are implemented below. Physical diagnostics run before the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import math
import csv
import platform
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import (
    ensure_dir,
    get_device,
    list_case_dirs,
    save_json,
    load_json,
    seed_everything,
    split_cases_by_ids,
)

from fno_model import FNO2dEnhanced, FNOConfig
from data_timewindow import (
    Batch,
    NormStats,
    PatchCorrectorWindowIterable,
    PatchTimeWindowIterable,
    compute_norm_stats,
)
from losses import (
    inner_mask,
    get_rho_from_alpha,
    interface_weight_from_alpha,
    masked_grad_mse,
    masked_mse,
    masked_rel_l2,
    masked_weighted_mse,
    poisson_consistency_loss,
    poisson_residual_loss,
    poisson_rAU_consistency_loss,
    poisson_rAU_residual_loss,
    spectral_mse,
    gauge_mean_anchor_loss,
)

from train_helpers import (
    _attach_learned_logvars,
    _clamp_learned_logvars,
    _logvar_weighted_term,
    _resolve_channel,
    _compute_gh,
    _rho_from_frame,
    _overwrite_prgh_if_needed,
    _overwrite_p_if_needed,
    build_x_from_window,
    y_phys_to_norm,
    y_norm_to_phys,
    _demean2d,
    _match_mean2d,
    _get_gauge_masks,
    _infer_dx_dy_from_invDx2,
    _grad2d,
    _pde_weight,
    _cont_weight,
    _div2d,
    _teacher_forcing_ratio,
    _teacher_forcing_ratio_step,
    _tout_for_epoch,
    _maybe_offset_augment,
    _set_optimizer_lr,
    _lr_for_step,
    _save_checkpoint,
    _export_libtorch_state_dict,
    _try_load_checkpoint
)

from train_checks import (
    poisson_alignment_check,
    continuity_sanity_check,
)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Mean absolute error over masked region (mask expected shape (B,1,H,W))."""
    m = mask.float()
    den = m.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    return ((pred - target).abs() * m).sum(dim=(2, 3), keepdim=True) / den
@dataclass
class TrainConfig:
    patch_root: Path
    out_dir: Path

    # dataset organization
    #  - "time": normal physical-time autoregression (t dimension)
    #  - "corrector": within one physical time step, autoregress over pressure-corrector k (k=0..K-1)
    dataset_mode: str = "corrector"

    # Optional supervision for velocity U (Ux,Uy), aligned by dp patch row indices.
    # Provide a root built by build_velocity_supervision.py.
    u_patch_root: Optional[Path] = None
    u_loss_weight: float = 0.0
    u_loss_final_only: bool = True


    # Continuity supervision (recommended over direct U-MSE): encourage corrected velocity to satisfy div(U)=0.
    # Requires u_patch_root, so we can build a proxy HbyA from (U_true, rAU, grad(p_rgh_true)).
    cont_loss_weight: float = 1e-4
    cont_warmup_epochs: int = 80
    cont_ramp_epochs: int = 320  # reaches the paper value at epoch 400
    cont_final_only: bool = True  # in corrector mode, only apply on the final step
    cont_interior_margin: Optional[int] = None  # default: reuse poisson_check_interior_margin
    cont_check_batches: int = 32
    do_continuity_sanity_check: bool = True

    # Learned log-variance multi-task weighting (optional).
    # If enabled, ONLY applies to "pde" and "cont" terms (dp stays the main objective with fixed weight).
    # Loss form per term i:  w_sched * (exp(-s_i) * L_i + s_i),  where s_i is a learned scalar (log variance).
    use_learned_logvars: bool = False
    learned_logvar_init: float = 0.0
    learned_logvar_min: float = -8.0
    learned_logvar_max: float = 8.0
    learned_logvar_tasks: Tuple[str, ...] = ("pde", "cont")

    # time window
    T_in: int = 1
    T_out: int = 3
    steps_per_epoch: int = 600

    # model (FNO)
    modes: int = 12
    width: int = 64
    padding: int = 12
    noise_std: float = 0.0

    # residual U-Net refinement head
    use_residual: bool = True
    residual_unet_depth: int = 4
    residual_unet_base_channels: int = 32
    residual_unet_gn_groups: int = 8
    residual_unet_dropout: float = 0.1

    # training
    epochs: int = 800
    batch_size: int = 16
    lr: float = 5e-4
    min_lr: float = 0.0
    lr_schedule: str = "warmup_cosine"  # options: warmup_cosine | constant
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    grad_clip: float = 1.0
    amp: bool = True
    cache_size: int = 8

    # checkpointing
    resume: bool = True
    resume_path: Optional[Path] = None  # default: out_dir/ckpt_last.pt
    save_ckpt: bool = True

    # export (for C++ libtorch: torch::load(net, "FNONet.pth"))
    export_libtorch_state_dict: bool = True
    export_libtorch_name: str = "FNONet.pth"
    export_libtorch_best_only: bool = True
    # scheduled sampling on p fed back into window
    tf_start: float = 1.0
    tf_end: float = 0.0
    tf_decay_portion: float = 0.7
    # step-wise teacher forcing: later rollout steps get less TF
    tf_step_gamma: float = 0.85
    tf_step_end: float = 0.0

    # curriculum on rollout horizon (train): start short then increase
    use_curriculum: bool = True
    curriculum_tout1: int = 1
    curriculum_epoch1: int = 50
    curriculum_tout2: int = 3
    curriculum_epoch2: int = 50
    eval_match_curriculum: bool = False

    # Poisson alignment sanity check (before training)
    do_poisson_alignment_check: bool = True
    poisson_check_batches: int = 32
    poisson_check_max_steps: int = 1
    poisson_check_interior_margin: int = 4

    # augmentation: random integer offset crop to reduce patch-boundary overfitting
    offset_aug_max: int = 16  # P/4 for the 64x64 training patches
    offset_aug_prob: float = 0.5

    # apply offset augmentation in validation/eval (default: False)
    offset_aug_in_val: bool = False
    # extra spectral loss on reconstructed next-field (p_next or p_rgh_next)
    spec_loss_weight: float = 3e-9
    spec_on_prgh: bool = True

    # what the network predicts: "p" means Δp, "prgh" means Δp_rgh
    # For the Poisson-input pipeline we recommend Δp_rgh.
    dp_target: str = "prgh"

    # channel indices WITHIN physical channels (X without last 3 grid channels)
    # Input order: [alpha, rhs, rAU, p_in, p_rgh_in, invDx2, invDy2].
    # allow negative indices (python-style)
    invrho_channel: Optional[int] = None  # legacy/optional: invRho = 1/rho_mix
    alpha_channel: int = 0  # alpha
    rhs_channel: Optional[int] = 1  # rhs for laplacian(rAUf, p_rgh) = rhs
    rAU_channel: Optional[int] = 2  # rAU (cell) for laplacian(rAUf, p_rgh)
    p_channel: int = 3  # p (used for reconstruct + feedback)
    prgh_channel: Optional[int] = 4  # p_rgh (optional input); recommended to keep
    invDx2_channel: Optional[int] = 5  # 1/dx^2 (optional; used by PDE loss if present)
    invDy2_channel: Optional[int] = 6  # 1/dy^2 (optional; used by PDE loss if present)

    # Normalize dynamic variables only. Static invDx2/invDy2 and x/y/mask
    # remain in their original representation, as reported in the supplement.
    normalize_channels: Tuple[int, ...] = (0, 1, 2, 3, 4)

    # During rollout we will overwrite one pressure-like channel and recompute the other so they remain
    # physically consistent via: p_rgh = p - rho*gh,  p = p_rgh + rho*gh.
    # If you rollout p (dp_target="p"), set recompute_prgh_in_rollout=True.
    # If you rollout p_rgh (dp_target="prgh"), set recompute_p_in_rollout=True.
    recompute_prgh_in_rollout: bool = True
    recompute_p_in_rollout: bool = True

    # geometry + gravity for p_rgh = p - rho*gh recompute
    # Recommended (current pipeline):
    # - store (x,y) normalized to [0,1] in X (last channels), plus mask
    # - store per-patch meta_json (x0,y0,Lx,Ly,dx,dy)
    # - data_timewindow returns grid (normalized) and grid_phys (meters)
    # IMPORTANT: trainer computes gh/PDE using grid_phys, so set grid_is_normalized=False
    # (If you want grid_is_normalized=True, then call _compute_gh with batch.grid, not batch.grid_phys.)
    domain_x0: float = 0.0
    domain_y0: float = 0.0
    domain_Lx: float = 1.0
    domain_Ly: float = 1.0
    g_x: float = 0.0
    g_y: float = -9.81
    gh_ref: float = 0.0  # OpenFOAM: gh = g·C - ghRef
    grid_is_normalized: bool = False  # x/y are already normalized in X; PDE uses grid_phys (meters) without extra scaling

    rho_channel: Optional[int] = None  # if you store rho_mix explicitly (rare)

    # mixture rho fallback (only used if invrho_channel and rho_channel are None)
    rho1: float = 1000.0  # liquid
    rho2: float = 1.2  # gas

    # loss enhancements
    dp_loss_weight: float = 1.0  # weight for the main supervised delta-p loss
    p_next_loss_weight: float = 0.0  # raw reconstruction loss is not part of the paper objective
    p_next_demean_loss_weight: float = 5e-2  # normalized, mean-centered reconstruction loss
    grad_loss_weight: float = 1e-2  # Sobel gradient loss on delta_p
    interface_lambda: float = 0.0  # interface weighting using alpha

    # PDE loss
    pde_loss_weight: float = 1e-8
    pde_use_rhs_if_available: bool = True
    pde_warmup_epochs: int = 120
    pde_ramp_epochs: int = 130  # reaches the paper value at epoch 250
    pde_normalize_by_rhs: bool = True
    pde_rhs_eps_abs: float = 1e-6
    pde_rhs_max_weight: float = 0.85
    pde_interior_margin: Optional[int] = None  # None => reuse poisson_check_interior_margin

    # Fixed case-level split from Table 1 in the paper.
    train_case_ids: Tuple[int, ...] = tuple(range(1, 11))
    val_case_ids: Tuple[int, ...] = (11,)
    test_case_ids: Tuple[int, ...] = (12, 13)

    # norm sampling
    max_norm_files_per_case: int = 6
    sample_patches_per_file: int = 128

    # misc
    seed: int = 0
    num_workers: int = 0


    # Gauge (pressure null-space) fixing
    gauge_loss_weight: float = 0.0
    gauge_interior_margin: Optional[int] = None  # None => reuse poisson_check_interior_margin
    gauge_demean_losses: bool = True  # remove constant mode in dp/p_next/spec losses
    gauge_match_mean_rollout: bool = False  # avoid using the reference mean during free rollout
    gauge_use_interior_mask: bool = True  # use inner_mask(mask, gauge_interior_margin) for gauge-related losses
    rel_metric_gauge_invariant: bool = True  # report relL2 after de-meaning pressure-like fields


def validate_config(cfg: TrainConfig) -> None:
    """Fail early when a run cannot reproduce the paper configuration."""
    if not cfg.patch_root.is_dir():
        raise FileNotFoundError(f"patch_root does not exist or is not a directory: {cfg.patch_root}")
    if cfg.dataset_mode != "corrector":
        raise ValueError("The paper configuration requires dataset_mode='corrector'")
    if cfg.T_in != 1 or cfg.T_out != 3:
        raise ValueError("The paper configuration requires T_in=1 and T_out=3")
    if cfg.dp_target != "prgh" or not cfg.spec_on_prgh:
        raise ValueError("The paper configuration predicts delta p_rgh and evaluates spectral loss in p_rgh space")
    if (
        cfg.alpha_channel,
        cfg.rhs_channel,
        cfg.rAU_channel,
        cfg.p_channel,
        cfg.prgh_channel,
        cfg.invDx2_channel,
        cfg.invDy2_channel,
    ) != (0, 1, 2, 3, 4, 5, 6):
        raise ValueError("Input channels must follow [alpha, rhs, rAU, p, p_rgh, invDx2, invDy2]")
    if tuple(cfg.normalize_channels) != (0, 1, 2, 3, 4):
        raise ValueError("Only the five dynamic physical channels may be normalized")
    if min(cfg.epochs, cfg.batch_size, cfg.steps_per_epoch) <= 0:
        raise ValueError("epochs, batch_size, and steps_per_epoch must be positive")
    paper_values = {
        "modes": 12,
        "width": 64,
        "padding": 12,
        "residual_unet_depth": 4,
        "residual_unet_base_channels": 32,
        "residual_unet_dropout": 0.1,
        "epochs": 800,
        "batch_size": 16,
        "steps_per_epoch": 600,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "warmup_epochs": 10,
        "tf_start": 1.0,
        "tf_end": 0.0,
        "tf_decay_portion": 0.7,
        "tf_step_gamma": 0.85,
        "offset_aug_max": 16,
        "offset_aug_prob": 0.5,
        "dp_loss_weight": 1.0,
        "grad_loss_weight": 1e-2,
        "p_next_loss_weight": 0.0,
        "p_next_demean_loss_weight": 5e-2,
        "spec_loss_weight": 3e-9,
        "pde_loss_weight": 1e-8,
        "cont_loss_weight": 1e-4,
        "pde_rhs_eps_abs": 1e-6,
        "pde_rhs_max_weight": 0.85,
    }
    mismatches = {
        name: (getattr(cfg, name), expected)
        for name, expected in paper_values.items()
        if getattr(cfg, name) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{name}={actual!r} (expected {expected!r})"
            for name, (actual, expected) in sorted(mismatches.items())
        )
        raise ValueError(f"Configuration differs from the paper: {details}")
    if cfg.cont_loss_weight > 0:
        if cfg.u_patch_root is None:
            raise ValueError(
                "Continuity loss is enabled by the paper configuration, so --u-patch-root is required"
            )
        if not cfg.u_patch_root.is_dir():
            raise FileNotFoundError(
                f"u_patch_root does not exist or is not a directory: {cfg.u_patch_root}"
            )
    if not (0.0 <= cfg.offset_aug_prob <= 1.0):
        raise ValueError("offset_aug_prob must be in [0, 1]")
    if cfg.offset_aug_max < 0:
        raise ValueError("offset_aug_max must be non-negative")
    if not (0.0 <= cfg.residual_unet_dropout < 1.0):
        raise ValueError("residual_unet_dropout must be in [0, 1)")
    if cfg.pde_warmup_epochs + cfg.pde_ramp_epochs != 250:
        raise ValueError("Poisson schedule must reach its final weight at epoch 250")
    if cfg.cont_warmup_epochs + cfg.cont_ramp_epochs != 400:
        raise ValueError("Continuity schedule must reach its final weight at epoch 400")
    if cfg.curriculum_epoch1 != 50 or cfg.curriculum_tout1 != 1:
        raise ValueError("The paper curriculum requires one-step training for the first 50 epochs")
    if cfg.tf_step_end != 0.0:
        raise ValueError("The paper teacher-forcing minimum probability is 0.0")
    if cfg.gauge_loss_weight != 0.0 or cfg.u_loss_weight != 0.0:
        raise ValueError("Gauge-anchor and direct velocity losses are not part of the paper objective")
    if cfg.gauge_match_mean_rollout:
        raise ValueError("Paper reproduction must not inject the reference mean into free rollout")



# =========================
# Helpers: normalization
# =========================

def train_one_epoch(
        model: torch.nn.Module,
        loader: DataLoader,
        opt: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler],
        device: torch.device,
        *,
        cfg: TrainConfig,
        norm: NormStats,
        epoch: int,
        state: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """
    AMP-safe training:
      - autocast ONLY wraps model forward()
      - ALL normalization + losses are forced to float32
    This prevents FP16 overflow (squared error / interface weights) -> NaN.
    """
    model.train()

    # keep norms in float32 (avoid accidental float64 inputs)
    x_mean = torch.from_numpy(norm.x_mean).to(device=device, dtype=torch.float32)
    x_std = torch.from_numpy(norm.x_std).to(device=device, dtype=torch.float32).clamp_min(1e-6)
    y_mean = torch.tensor(norm.y_mean, device=device, dtype=torch.float32).view(1, 1, 1, 1)
    y_std = torch.tensor(norm.y_std, device=device, dtype=torch.float32).view(1, 1, 1, 1).clamp_min(1e-6)

    autocast_enabled = bool(cfg.amp and device.type == "cuda")
    pde_w_sched = _pde_weight(cfg, epoch)
    cont_w_sched = _cont_weight(cfg, epoch)

    loss_sum = 0.0
    n_batches = 0

    # ---- NEW: component accumulators (per rollout step) ----
    comp_sum = {"dp": 0.0, "grad": 0.0, "pnext": 0.0, "pnext_dm": 0.0, "spec": 0.0, "gauge": 0.0, "pde": 0.0, "cont": 0.0, "u": 0.0}
    comp_w_sum = {"dp_w": 0.0, "grad_w": 0.0, "pnext_w": 0.0, "pnext_dm_w": 0.0, "spec_w": 0.0, "gauge_w": 0.0, "pde_w": 0.0, "cont_w": 0.0, "u_w": 0.0}
    n_terms = 0

    Tout_use = _tout_for_epoch(cfg, int(epoch))
    for batch in loader:
        # per-step LR schedule (warmup + cosine)
        step_i = int(state.get('global_step', 0))
        total_steps = int(state.get('total_steps', 1))
        warmup_steps = int(state.get('warmup_steps', 0))
        lr_now = _lr_for_step(cfg, step_i, total_steps, warmup_steps)
        _set_optimizer_lr(opt, lr_now)

        n_batches += 1
        batch = Batch(
            phys_hist=batch.phys_hist.to(device),
            phys_fut=batch.phys_fut.to(device),
            dp_fut=batch.dp_fut.to(device),
            grid=batch.grid.to(device),
            grid_phys=batch.grid_phys.to(device),
            u_true=(batch.u_true.to(device) if getattr(batch, "u_true", None) is not None else None),
        )

        # augmentation: random offset crop (helps reduce block artifacts)
        if cfg.offset_aug_max > 0 and cfg.offset_aug_prob > 0:
            batch = _maybe_offset_augment(batch, max_shift=int(cfg.offset_aug_max), prob=float(cfg.offset_aug_prob))

        B, Tin, n_phys, H, W = batch.phys_hist.shape
        alpha_ch = _resolve_channel(cfg.alpha_channel, n_phys)
        p_ch = _resolve_channel(cfg.p_channel, n_phys)
        invrho_ch = _resolve_channel(cfg.invrho_channel, n_phys) if cfg.invrho_channel is not None else None
        rho_ch = _resolve_channel(cfg.rho_channel, n_phys) if cfg.rho_channel is not None else None
        rhs_ch = _resolve_channel(cfg.rhs_channel, n_phys) if cfg.rhs_channel is not None else None
        prgh_ch = _resolve_channel(cfg.prgh_channel, n_phys) if cfg.prgh_channel is not None else None
        rAU_ch = _resolve_channel(cfg.rAU_channel, n_phys) if cfg.rAU_channel is not None else None
        invDx2_ch = _resolve_channel(cfg.invDx2_channel, n_phys) if cfg.invDx2_channel is not None else None
        invDy2_ch = _resolve_channel(cfg.invDy2_channel, n_phys) if cfg.invDy2_channel is not None else None

        grid_norm = batch.grid
        grid_phys = batch.grid_phys
        mask = grid_norm[:, 2:3].float()


        # gauge masks for pressure-like fields (mean anchor + demean)
        mean_mask, loss_mask = _get_gauge_masks(cfg, mask)
        phys_win = batch.phys_hist  # (B,T_in,n_phys,H,W)
        phys_fut = batch.phys_fut  # (B,T_out-1,n_phys,H,W)
        dp_fut = batch.dp_fut  # (B,T_out,1,H,W)

        opt.zero_grad(set_to_none=True)
        step_losses = []

        # dp_target controls how we reconstruct next state from the predicted delta
        dp_target = str(getattr(cfg, "dp_target", "prgh")).lower()
        if dp_target.startswith("prgh") and prgh_ch is None:
            raise ValueError("cfg.dp_target='prgh' requires cfg.prgh_channel to be set")
        if dp_target.startswith("p") and p_ch is None:
            raise ValueError("cfg.dp_target='p' requires cfg.p_channel to be set")

        for t in range(Tout_use):
            tf_ratio_t = _teacher_forcing_ratio_step(cfg, epoch, t)
            true_frame_in = (
                batch.phys_hist[:, -1].float()
                if t == 0
                else phys_fut[:, t - 1].float()
            )

            # -------- FP32: build input + normalize target
            x = build_x_from_window(
                phys_win,
                grid_norm,
                x_mean=x_mean,
                x_std=x_std,
                T_in=cfg.T_in,
                normalize_channels=cfg.normalize_channels,
            ).float()

            dp_true_phys = dp_fut[:, t].float()  # physical delta_p
            dp_true_norm = y_phys_to_norm(dp_true_phys, y_mean=y_mean, y_std=y_std).float()

            # interface weights (FP32)
            if cfg.interface_lambda > 0:
                alpha_for_w = phys_win[:, -1, alpha_ch:alpha_ch + 1].float()
                w_int = interface_weight_from_alpha(
                    alpha_for_w,
                    interface_lambda=float(cfg.interface_lambda),
                ).float()
            else:
                w_int = None

            # -------- AMP: model forward only
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
                dp_pred_norm = model(x)  # normalized delta_p (maybe fp16)
            # -------- FP32: dp loss (optionally gauge-invariant)
            dp_pred_norm_f = dp_pred_norm.float()
            dp_pred_phys = y_norm_to_phys(dp_pred_norm_f, y_mean=y_mean, y_std=y_std).float()

            if bool(getattr(cfg, "gauge_demean_losses", True)):
                dp_pred_phys_l = _demean2d(dp_pred_phys, mean_mask)
                dp_true_phys_l = _demean2d(dp_true_phys, mean_mask)
                dp_pred_norm_l = y_phys_to_norm(dp_pred_phys_l, y_mean=y_mean, y_std=y_std).float()
                dp_true_norm_l = y_phys_to_norm(dp_true_phys_l, y_mean=y_mean, y_std=y_std).float()
            else:
                dp_pred_norm_l = dp_pred_norm_f
                dp_true_norm_l = dp_true_norm

            if w_int is None:
                dp_loss = masked_mse(dp_pred_norm_l, dp_true_norm_l, loss_mask)
            else:
                dp_loss = masked_weighted_mse(dp_pred_norm_l, dp_true_norm_l, loss_mask, w_int)
            dp_w = float(getattr(cfg, "dp_loss_weight", 1.0))


            # -------- FP32: optional grad loss (physical)
            grad_loss = torch.zeros((), device=device, dtype=torch.float32)
            if cfg.grad_loss_weight > 0:
                grad_loss = masked_grad_mse(dp_pred_phys, dp_true_phys, mask, w_int)

            # reconstruct next state (physical) from predicted delta
            gh = _compute_gh(cfg, grid_phys.float())
            rho_prev = _rho_from_frame(
                cfg,
                frame=phys_win[:, -1].float(),
                alpha_ch=alpha_ch,
                rho_ch=rho_ch,
                invrho_ch=invrho_ch,
            )

            if dp_target.startswith("prgh"):
                prgh_prev_phys = phys_win[:, -1, prgh_ch:prgh_ch + 1].float()
                p_rgh_pred = prgh_prev_phys + dp_pred_phys
                p_next_pred = p_rgh_pred + rho_prev * gh
            else:
                p_prev_phys = phys_win[:, -1, p_ch:p_ch + 1].float()
                p_next_pred = p_prev_phys + dp_pred_phys
                p_rgh_pred = p_next_pred - rho_prev * gh

            # -------- FP32: optional p_next supervision + spectral loss
            pnext_loss = torch.zeros((), device=device, dtype=torch.float32)
            pnext_demean_loss = torch.zeros((), device=device, dtype=torch.float32)
            spec_loss = torch.zeros((), device=device, dtype=torch.float32)

            gauge_loss = torch.zeros((), device=device, dtype=torch.float32)
            has_next = (t < (Tout_use - 1)) and (t < phys_fut.shape[1])
            if dp_target.startswith("prgh"):
                next_true = true_frame_in[:, prgh_ch:prgh_ch + 1] + dp_true_phys
                next_pred = p_rgh_pred
            else:
                next_true = true_frame_in[:, p_ch:p_ch + 1] + dp_true_phys
                next_pred = p_next_pred

            w_gauge = float(getattr(cfg, 'gauge_loss_weight', 0.0))
            if w_gauge > 0.0:
                gauge_loss = gauge_mean_anchor_loss(next_pred, next_true, mean_mask)
            pnext_loss = masked_mse(next_pred, next_true, loss_mask)
            rec_ch = prgh_ch if dp_target.startswith("prgh") else p_ch
            rec_std = x_std[rec_ch].view(1, 1, 1, 1).clamp_min(1e-12)
            pnext_demean_loss = masked_mse(
                _demean2d(next_pred, mean_mask) / rec_std,
                _demean2d(next_true, mean_mask) / rec_std,
                loss_mask,
            )

            if cfg.spec_loss_weight > 0:
                spec_loss = spectral_mse(
                    _demean2d(next_pred, mean_mask),
                    _demean2d(next_true, mean_mask),
                    loss_mask,
                )

            # -------- FP32: optional PDE loss (Poisson on p_rgh)
            pde_loss = torch.zeros((), device=device, dtype=torch.float32)
            pde_active = False
            if pde_w_sched > 0:
                pde_active = True
                # The pressure increment is produced by the equation assembled
                # from the current corrector input, so use the same-step rhs/rAU.
                rho_next = _rho_from_frame(cfg, frame=true_frame_in, alpha_ch=alpha_ch, rho_ch=rho_ch, invrho_ch=invrho_ch)

                if rAU_ch is not None:
                    rAU = true_frame_in[:, rAU_ch:rAU_ch + 1]
                    invDx2 = true_frame_in[:, invDx2_ch:invDx2_ch + 1] if invDx2_ch is not None else None
                    invDy2 = true_frame_in[:, invDy2_ch:invDy2_ch + 1] if invDy2_ch is not None else None

                    if rhs_ch is not None and cfg.pde_use_rhs_if_available:
                        rhs_f = true_frame_in[:, rhs_ch:rhs_ch + 1]
                        pde_loss = poisson_rAU_residual_loss(
                            p_rgh_pred, rAU, rhs_f, grid_phys.float(),
                            invDx2=invDx2, invDy2=invDy2,
                            normalize_by_rhs=bool(cfg.pde_normalize_by_rhs),
                            rhs_eps_abs=float(cfg.pde_rhs_eps_abs),
                            rhs_max_weight=float(cfg.pde_rhs_max_weight),
                            interior_margin=int(getattr(cfg, 'poisson_check_interior_margin', 1)),
                        )
                    else:
                        if prgh_ch is not None:
                            p_rgh_true = next_true
                        else:
                            p_next_true = next_true
                            p_rgh_true = p_next_true - rho_next * gh
                        pde_loss = poisson_rAU_consistency_loss(
                            p_rgh_pred, p_rgh_true, rAU, grid_phys.float(), invDx2=invDx2, invDy2=invDy2,
                            interior_margin=int(getattr(cfg, 'poisson_check_interior_margin', 1))
                        )
                else:
                    # Legacy fallback (older dataset): div((1/rho) grad(p)) consistency/residual
                    if rhs_ch is not None and cfg.pde_use_rhs_if_available:
                        rhs_f = true_frame_in[:, rhs_ch:rhs_ch + 1]
                        pde_loss = poisson_residual_loss(p_next_pred, rho_next, rhs_f, grid_phys.float(), interior_margin=int(getattr(cfg, 'poisson_check_interior_margin', 1)))
                    else:
                        p_next_true = (
                            next_true
                            if not dp_target.startswith("prgh")
                            else next_true + rho_next * gh
                        )
                        pde_loss = poisson_consistency_loss(p_next_pred, p_next_true, rho_next, grid_phys.float(), interior_margin=int(getattr(cfg, 'poisson_check_interior_margin', 1)))
            # -------- optional continuity supervision (recommended over direct U-MSE)
            # Encourage corrected velocity to satisfy div(U)=0 on the *interior* of the patch.
            cont_loss = torch.zeros((), device=device, dtype=torch.float32)
            w_cont = float(cont_w_sched)
            cont_active = False
            if (w_cont > 0.0) and getattr(batch, "u_true", None) is not None:
                do_cont = True
                if getattr(cfg, "cont_final_only", True):
                    do_cont = (t == (Tout_use - 1)) and (Tout_use == cfg.T_out)

                if do_cont and dp_target.startswith("prgh") and (rAU_ch is not None) and (invDx2_ch is not None) and (invDy2_ch is not None):
                    frame_u = (phys_fut[:, -1] if phys_fut.shape[1] > 0 else phys_win[:, -1]).float()
                    rAU_f = frame_u[:, rAU_ch:rAU_ch + 1].float()
                    invDx2_f = frame_u[:, invDx2_ch:invDx2_ch + 1].float()
                    invDy2_f = frame_u[:, invDy2_ch:invDy2_ch + 1].float()
                    dx, dy = _infer_dx_dy_from_invDx2(invDx2_f, invDy2_f)

                    prgh_true_in_f = frame_u[:, prgh_ch:prgh_ch + 1].float()
                    prgh_true_final = prgh_true_in_f + dp_fut[:, -1].float()

                    dpdx_true, dpdy_true = _grad2d(prgh_true_final, dx, dy)
                    dpdx_pred, dpdy_pred = _grad2d(p_rgh_pred.float(), dx, dy)

                    U_true = batch.u_true.float()  # (B,2,H,W)
                    Ux_true = U_true[:, 0:1]
                    Uy_true = U_true[:, 1:2]

                    HbyA_x = Ux_true + rAU_f * dpdx_true
                    HbyA_y = Uy_true + rAU_f * dpdy_true
                    Ux_pred = HbyA_x - rAU_f * dpdx_pred
                    Uy_pred = HbyA_y - rAU_f * dpdy_pred

                    div_pred = _div2d(Ux_pred, Uy_pred, dx, dy)

                    margin = getattr(cfg, "cont_interior_margin", None)
                    if margin is None:
                        margin = int(getattr(cfg, "poisson_check_interior_margin", 1))
                    m_int = inner_mask(mask.float(), margin=int(margin))
                    cont_active = True
                    cont_loss = masked_mse(div_pred, torch.zeros_like(div_pred), m_int)

            # -------- optional U supervision
            # This term reconstructs U from pressure-gradient correction
            # and compares it with OpenFOAM U_true.
            u_loss = torch.zeros((), device=device, dtype=torch.float32)
            w_u = float(getattr(cfg, "u_loss_weight", 0.0))

            if w_u > 0.0 and getattr(batch, "u_true", None) is not None:
                do_u = True

                # By default, only apply U supervision at the final corrector step.
                if bool(getattr(cfg, "u_loss_final_only", True)):
                    do_u = (t == (Tout_use - 1)) and (Tout_use == cfg.T_out)

                if (
                        do_u
                        and dp_target.startswith("prgh")
                        and (rAU_ch is not None)
                        and (invDx2_ch is not None)
                        and (invDy2_ch is not None)
                        and (prgh_ch is not None)
                ):
                    # Use the final corrector-related frame to build the velocity correction.
                    frame_u = (phys_fut[:, -1] if phys_fut.shape[1] > 0 else phys_win[:, -1]).float()

                    rAU_f = frame_u[:, rAU_ch:rAU_ch + 1].float()
                    invDx2_f = frame_u[:, invDx2_ch:invDx2_ch + 1].float()
                    invDy2_f = frame_u[:, invDy2_ch:invDy2_ch + 1].float()

                    dx, dy = _infer_dx_dy_from_invDx2(invDx2_f, invDy2_f)

                    # True final p_rgh at the final corrector step:
                    # p_rgh_final_true = p_rgh_input + delta_p_rgh_true
                    prgh_true_in_f = frame_u[:, prgh_ch:prgh_ch + 1].float()
                    prgh_true_final = prgh_true_in_f + dp_fut[:, -1].float()

                    # Pressure gradients from true and predicted p_rgh
                    dpdx_true, dpdy_true = _grad2d(prgh_true_final, dx, dy)
                    dpdx_pred, dpdy_pred = _grad2d(p_rgh_pred.float(), dx, dy)

                    # U_true comes from u_patch_root, shape: (B, 2, H, W)
                    U_true = batch.u_true.float()
                    Ux_true = U_true[:, 0:1]
                    Uy_true = U_true[:, 1:2]

                    # Reconstruct HbyA using the true velocity and true pressure gradient:
                    # U_true = HbyA - rAU * grad(p_rgh_true)
                    # so HbyA = U_true + rAU * grad(p_rgh_true)
                    HbyA_x = Ux_true + rAU_f * dpdx_true
                    HbyA_y = Uy_true + rAU_f * dpdy_true

                    # Predicted corrected velocity:
                    # U_pred = HbyA - rAU * grad(p_rgh_pred)
                    Ux_pred = HbyA_x - rAU_f * dpdx_pred
                    Uy_pred = HbyA_y - rAU_f * dpdy_pred

                    U_pred = torch.cat([Ux_pred, Uy_pred], dim=1)

                    # Compute masked MSE over Ux and Uy together
                    Bc, _, Hh, Ww = U_pred.shape
                    u_pred_flat = U_pred.reshape(Bc * 2, 1, Hh, Ww)
                    u_true_flat = U_true.reshape(Bc * 2, 1, Hh, Ww)
                    mask_flat = mask.repeat_interleave(2, dim=0)

                    u_loss = masked_mse(u_pred_flat, u_true_flat, mask_flat)

            # -------- total loss (match training objective)
            w_grad = float(cfg.grad_loss_weight)
            w_pnext = float(cfg.p_next_loss_weight)
            w_pnext_dm = float(getattr(cfg, "p_next_demean_loss_weight", 0.0))
            w_spec = float(cfg.spec_loss_weight)
            w_pde = float(pde_w_sched)
            w_cont = float(cont_w_sched)

            loss_t = (
                    dp_w * dp_loss
                    + w_grad * grad_loss
                    + w_pnext * pnext_loss
                    + w_pnext_dm * pnext_demean_loss
                    + w_spec * spec_loss
                    + float(getattr(cfg, 'gauge_loss_weight', 0.0)) * gauge_loss
            )# learned log-variance weighting for auxiliary terms (PDE + continuity)
            use_logvars = bool(getattr(cfg, "use_learned_logvars", False)) and hasattr(model, "loss_logvars")

            if pde_active:
                if use_logvars and ("pde" in model.loss_logvars):  # type: ignore[attr-defined]
                    s_pde = model.loss_logvars["pde"]  # type: ignore[attr-defined]
                    loss_t = loss_t + w_pde * _logvar_weighted_term(pde_loss, s_pde)
                else:
                    loss_t = loss_t + w_pde * pde_loss

            if cont_active:
                if use_logvars and ("cont" in model.loss_logvars):  # type: ignore[attr-defined]
                    s_cont = model.loss_logvars["cont"]  # type: ignore[attr-defined]
                    loss_t = loss_t + w_cont * _logvar_weighted_term(cont_loss, s_cont)
                else:
                    loss_t = loss_t + w_cont * cont_loss

            # Keep optional U-MSE term as-is (if you still enable it).
            w_u = float(getattr(cfg, "u_loss_weight", 0.0))
            if w_u > 0.0:
                loss_t = loss_t + w_u * u_loss

            step_losses.append(loss_t)

            # ---- NEW: log accumulators (raw + weighted contributions)
            comp_sum["dp"] += float(dp_loss.detach().cpu())
            comp_sum["grad"] += float(grad_loss.detach().cpu())
            comp_sum["pnext"] += float(pnext_loss.detach().cpu())
            comp_sum["pnext_dm"] += float(pnext_demean_loss.detach().cpu())
            comp_sum["spec"] += float(spec_loss.detach().cpu())
            comp_sum["gauge"] += float(gauge_loss.detach().cpu())
            comp_sum["pde"] += float(pde_loss.detach().cpu())
            comp_sum["cont"] += float(cont_loss.detach().cpu())
            comp_sum["u"] += float(u_loss.detach().cpu())

            comp_w_sum["dp_w"] += float((dp_w * dp_loss).detach().cpu())
            comp_w_sum["grad_w"] += float((w_grad * grad_loss).detach().cpu())
            comp_w_sum["pnext_w"] += float((w_pnext * pnext_loss).detach().cpu())
            comp_w_sum["pnext_dm_w"] += float((w_pnext_dm * pnext_demean_loss).detach().cpu())
            comp_w_sum["spec_w"] += float((w_spec * spec_loss).detach().cpu())
            comp_w_sum["gauge_w"] += float((float(getattr(cfg, "gauge_loss_weight", 0.0)) * gauge_loss).detach().cpu())
            # Weighted contributions: if using learned logvars, log the *actual* added term (w*(exp(-s)*L + s)).
            use_logvars = bool(getattr(cfg, "use_learned_logvars", False)) and hasattr(model, "loss_logvars")

            if pde_active and use_logvars and ("pde" in model.loss_logvars):  # type: ignore[attr-defined]
                s_pde = model.loss_logvars["pde"]  # type: ignore[attr-defined]
                pde_term = w_pde * _logvar_weighted_term(pde_loss, s_pde)
                comp_w_sum["pde_w"] += float(pde_term.detach().cpu())
                comp_sum["pde_logv"] = comp_sum.get("pde_logv", 0.0) + float(s_pde.detach().cpu())
                comp_sum["pde_w_eff"] = comp_sum.get("pde_w_eff", 0.0) + float((torch.exp(-s_pde) * w_pde).detach().cpu())
            else:
                comp_w_sum["pde_w"] += float((w_pde * pde_loss).detach().cpu())

            if cont_active and use_logvars and ("cont" in model.loss_logvars):  # type: ignore[attr-defined]
                s_cont = model.loss_logvars["cont"]  # type: ignore[attr-defined]
                cont_term = w_cont * _logvar_weighted_term(cont_loss, s_cont)
                comp_w_sum["cont_w"] += float(cont_term.detach().cpu())
                comp_sum["cont_logv"] = comp_sum.get("cont_logv", 0.0) + float(s_cont.detach().cpu())
                comp_sum["cont_w_eff"] = comp_sum.get("cont_w_eff", 0.0) + float((torch.exp(-s_cont) * w_cont).detach().cpu())
            else:
                comp_w_sum["cont_w"] += float((w_cont * cont_loss).detach().cpu())
            comp_w_sum["u_w"] += float((w_u * u_loss).detach().cpu())

            n_terms += 1

            # shift window (except last step)
            if t < Tout_use - 1:
                next_frame = phys_fut[:, t].clone()

                if dp_target.startswith("prgh"):
                    prgh_next_true = phys_fut[:, t, prgh_ch:prgh_ch + 1].float()
                    prgh_next_pred_for_feed = p_rgh_pred
                    if bool(getattr(cfg, 'gauge_match_mean_rollout', True)):
                        prgh_next_pred_for_feed = _match_mean2d(prgh_next_pred_for_feed, prgh_next_true, mean_mask)
                    # teacher forcing mix
                    if tf_ratio_t >= 1.0:
                        prgh_next_feed = prgh_next_true
                    elif tf_ratio_t <= 0.0:
                        prgh_next_feed = prgh_next_pred_for_feed
                    else:
                        use_tf = (torch.rand((B, 1, 1, 1), device=device) < tf_ratio_t)
                        prgh_next_feed = torch.where(use_tf, prgh_next_true, prgh_next_pred_for_feed)

                    next_frame[:, prgh_ch:prgh_ch + 1] = prgh_next_feed
                    _overwrite_p_if_needed(
                        cfg,
                        next_frame=next_frame,
                        prgh_for_p=prgh_next_feed,
                        grid=grid_phys,
                        alpha_ch=alpha_ch,
                        rho_ch=rho_ch,
                        invrho_ch=invrho_ch,
                        p_ch=p_ch,
                    )
                else:
                    p_next_true = phys_fut[:, t, p_ch:p_ch + 1].float()
                    p_next_pred_for_feed = p_next_pred
                    if bool(getattr(cfg, "gauge_match_mean_rollout", True)):
                        p_next_pred_for_feed = _match_mean2d(p_next_pred_for_feed, p_next_true, mean_mask)
                    if tf_ratio_t >= 1.0:
                        p_next_feed = p_next_true
                    elif tf_ratio_t <= 0.0:
                        p_next_feed = p_next_pred_for_feed
                    else:
                        use_tf = (torch.rand((B, 1, 1, 1), device=device) < tf_ratio_t)
                        p_next_feed = torch.where(use_tf, p_next_true, p_next_pred_for_feed)

                    next_frame[:, p_ch:p_ch + 1] = p_next_feed
                    _overwrite_prgh_if_needed(
                        cfg,
                        next_frame=next_frame,
                        p_for_prgh=p_next_feed,
                        grid=grid_phys,
                        alpha_ch=alpha_ch,
                        rho_ch=rho_ch,
                        invrho_ch=invrho_ch,
                        prgh_ch=prgh_ch,
                    )

                phys_win = torch.cat([phys_win[:, 1:], next_frame.unsqueeze(1)], dim=1)

        loss = torch.stack(step_losses).mean()

        # early fail fast
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss detected (epoch={epoch}). "
                f"Try amp=False, or reduce interface_lambda / grad_loss_weight, "
                f"or check norm_stats + scaling."
            )

        if scaler is not None and autocast_enabled:
            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            scaler.step(opt)
            scaler.update()

            if bool(getattr(cfg, 'use_learned_logvars', False)):
                _clamp_learned_logvars(model, cfg)
        else:
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            opt.step()

            if bool(getattr(cfg, 'use_learned_logvars', False)):
                _clamp_learned_logvars(model, cfg)

        loss_sum += float(loss.detach().cpu())
        state['global_step'] = step_i + 1

    # ---- NEW: return averaged component dict ----
    denom = max(n_terms, 1)
    comp_avg = {k: v / denom for k, v in comp_sum.items()}
    comp_w_avg = {k: v / denom for k, v in comp_w_sum.items()}

    return loss_sum / max(n_batches, 1), {**comp_avg, **comp_w_avg}


@torch.no_grad()
def eval_one_epoch(
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        *,
        cfg: TrainConfig,
        norm: NormStats,
        epoch: int = 0,
) -> Tuple[float, float, Dict[str, float]]:
    model.eval()

    x_mean = torch.from_numpy(norm.x_mean).to(device=device, dtype=torch.float32)
    x_std = torch.from_numpy(norm.x_std).to(device=device, dtype=torch.float32).clamp_min(1e-6)
    y_mean = torch.tensor(norm.y_mean, device=device, dtype=torch.float32).view(1, 1, 1, 1)
    y_std = torch.tensor(norm.y_std, device=device, dtype=torch.float32).view(1, 1, 1, 1).clamp_min(1e-6)

    pde_w_sched = _pde_weight(cfg, epoch)
    cont_w_sched = _cont_weight(cfg, epoch)

    loss_sum = 0.0
    rel_sum = 0.0
    rollout_rel_sum = 0.0
    rollout_mae_sum = 0.0
    rollout_pde_sum = 0.0
    n_batches = 0

    # ---- NEW: component accumulators (per rollout step) ----
    comp_sum = {"dp": 0.0, "grad": 0.0, "pnext": 0.0, "pnext_dm": 0.0, "spec": 0.0, "gauge": 0.0, "pde": 0.0, "cont": 0.0, "u": 0.0, "rollout_rel": 0.0, "rollout_mae": 0.0, "rollout_pde": 0.0}
    comp_w_sum = {"dp_w": 0.0, "grad_w": 0.0, "pnext_w": 0.0, "pnext_dm_w": 0.0, "spec_w": 0.0, "gauge_w": 0.0, "pde_w": 0.0, "cont_w": 0.0, "u_w": 0.0}
    n_terms = 0

    Tout_use = cfg.T_out
    if bool(getattr(cfg, "eval_match_curriculum", False)):
        Tout_use = _tout_for_epoch(cfg, int(epoch))

    for batch in loader:
        n_batches += 1
        batch = Batch(
            phys_hist=batch.phys_hist.to(device),
            phys_fut=batch.phys_fut.to(device),
            dp_fut=batch.dp_fut.to(device),
            grid=batch.grid.to(device),
            grid_phys=batch.grid_phys.to(device),
            u_true=(batch.u_true.to(device) if getattr(batch, "u_true", None) is not None else None),
        )

        if cfg.offset_aug_max > 0 and cfg.offset_aug_prob > 0 and bool(getattr(cfg, 'offset_aug_in_val', False)):
            batch = _maybe_offset_augment(batch, max_shift=int(cfg.offset_aug_max), prob=float(cfg.offset_aug_prob))

        B, Tin, n_phys, H, W = batch.phys_hist.shape
        alpha_ch = _resolve_channel(cfg.alpha_channel, n_phys)
        p_ch = _resolve_channel(cfg.p_channel, n_phys)

        invrho_ch = _resolve_channel(cfg.invrho_channel, n_phys) if cfg.invrho_channel is not None else None
        rho_ch = _resolve_channel(cfg.rho_channel, n_phys) if cfg.rho_channel is not None else None
        rhs_ch = _resolve_channel(cfg.rhs_channel, n_phys) if cfg.rhs_channel is not None else None
        prgh_ch = _resolve_channel(cfg.prgh_channel, n_phys) if cfg.prgh_channel is not None else None
        rAU_ch = _resolve_channel(cfg.rAU_channel, n_phys) if cfg.rAU_channel is not None else None
        invDx2_ch = _resolve_channel(cfg.invDx2_channel, n_phys) if cfg.invDx2_channel is not None else None
        invDy2_ch = _resolve_channel(cfg.invDy2_channel, n_phys) if cfg.invDy2_channel is not None else None

        grid_norm = batch.grid
        grid_phys = batch.grid_phys
        mask = grid_norm[:, 2:3].float()


        # gauge masks for pressure-like fields (mean anchor + demean)
        mean_mask, loss_mask = _get_gauge_masks(cfg, mask)
        phys_win = batch.phys_hist
        phys_fut = batch.phys_fut
        dp_fut = batch.dp_fut

        step_losses = []
        step_rels = []

        dp_target = str(getattr(cfg, "dp_target", "prgh")).lower()
        if dp_target.startswith("prgh") and prgh_ch is None:
            raise RuntimeError("dp_target='prgh' requires prgh_channel")

        gh = _compute_gh(cfg, grid_phys.float())

        for t in range(Tout_use):
            true_frame_in = (
                batch.phys_hist[:, -1].float()
                if t == 0
                else phys_fut[:, t - 1].float()
            )
            x = build_x_from_window(
                phys_win,
                grid_norm,
                x_mean=x_mean,
                x_std=x_std,
                T_in=cfg.T_in,
                normalize_channels=cfg.normalize_channels,
            ).float()
            dp_pred_norm = model(x).float()

            dp_true_phys = dp_fut[:, t].float()
            dp_true_norm = y_phys_to_norm(dp_true_phys, y_mean=y_mean, y_std=y_std).float()

            # interface weights (keep consistent with train)
            if cfg.interface_lambda > 0:
                alpha_for_w = phys_win[:, -1, alpha_ch:alpha_ch + 1].float()
                w_int = interface_weight_from_alpha(alpha_for_w, interface_lambda=float(cfg.interface_lambda)).float()
            else:
                w_int = None
            # dp loss (optionally gauge-invariant)
            dp_pred_phys = y_norm_to_phys(dp_pred_norm, y_mean=y_mean, y_std=y_std).float()

            if bool(getattr(cfg, "gauge_demean_losses", True)):
                dp_pred_phys_l = _demean2d(dp_pred_phys, mean_mask)
                dp_true_phys_l = _demean2d(dp_true_phys, mean_mask)
                dp_pred_norm_l = y_phys_to_norm(dp_pred_phys_l, y_mean=y_mean, y_std=y_std).float()
                dp_true_norm_l = y_phys_to_norm(dp_true_phys_l, y_mean=y_mean, y_std=y_std).float()
            else:
                dp_pred_norm_l = dp_pred_norm
                dp_true_norm_l = dp_true_norm

            if w_int is None:
                dp_loss = masked_mse(dp_pred_norm_l, dp_true_norm_l, loss_mask)
            else:
                dp_loss = masked_weighted_mse(dp_pred_norm_l, dp_true_norm_l, loss_mask, w_int)
            dp_w = float(getattr(cfg, "dp_loss_weight", 1.0))


            # grad loss
            grad_loss = torch.zeros((), device=device, dtype=torch.float32)
            if cfg.grad_loss_weight > 0:
                grad_loss = masked_grad_mse(dp_pred_phys, dp_true_phys, mask, w_int)

            # reconstruct predicted next field
            rho_prev = _rho_from_frame(
                cfg,
                frame=phys_win[:, -1].float(),
                alpha_ch=alpha_ch,
                rho_ch=rho_ch,
                invrho_ch=invrho_ch,
            )

            if dp_target.startswith("prgh"):
                prgh_prev = phys_win[:, -1, prgh_ch:prgh_ch + 1].float()
                prgh_next_pred = prgh_prev + dp_pred_phys
                p_next_pred = prgh_next_pred + rho_prev * gh
            else:
                p_prev = phys_win[:, -1, p_ch:p_ch + 1].float()
                p_next_pred = p_prev + dp_pred_phys
                prgh_next_pred = p_next_pred - rho_prev * gh

            has_next = (t < (Tout_use - 1)) and (t < phys_fut.shape[1])
            if dp_target.startswith("prgh"):
                next_true = true_frame_in[:, prgh_ch:prgh_ch + 1] + dp_true_phys
                next_pred = prgh_next_pred
            else:
                next_true = true_frame_in[:, p_ch:p_ch + 1] + dp_true_phys
                next_pred = p_next_pred

            rel_t = masked_rel_l2(
                _demean2d(next_pred, mean_mask)
                if bool(getattr(cfg, "rel_metric_gauge_invariant", True))
                else next_pred,
                _demean2d(next_true, mean_mask)
                if bool(getattr(cfg, "rel_metric_gauge_invariant", True))
                else next_true,
                loss_mask,
            )

            pnext_loss = torch.zeros((), device=device, dtype=torch.float32)
            pnext_demean_loss = torch.zeros((), device=device, dtype=torch.float32)
            spec_loss = torch.zeros((), device=device, dtype=torch.float32)
            pde_loss = torch.zeros((), device=device, dtype=torch.float32)
            cont_loss = torch.zeros((), device=device, dtype=torch.float32)
            gauge_loss = torch.zeros((), device=device, dtype=torch.float32)
            w_gauge = float(getattr(cfg, "gauge_loss_weight", 0.0))
            if w_gauge > 0.0:
                gauge_loss = gauge_mean_anchor_loss(next_pred, next_true, mean_mask)
            pnext_loss = masked_mse(next_pred, next_true, loss_mask)
            rec_ch = prgh_ch if dp_target.startswith("prgh") else p_ch
            rec_std = x_std[rec_ch].view(1, 1, 1, 1).clamp_min(1e-12)
            pnext_demean_loss = masked_mse(
                _demean2d(next_pred, mean_mask) / rec_std,
                _demean2d(next_true, mean_mask) / rec_std,
                loss_mask,
            )

            if cfg.spec_loss_weight > 0:
                spec_loss = spectral_mse(
                    _demean2d(next_pred, mean_mask),
                    _demean2d(next_true, mean_mask),
                    loss_mask,
                )

            # Validation always reports the raw PDE term; only its weighted
            # contribution follows the warmup/ramp schedule.
            rho_next = _rho_from_frame(
                cfg,
                frame=true_frame_in,
                alpha_ch=alpha_ch,
                rho_ch=rho_ch,
                invrho_ch=invrho_ch,
            )
            if rAU_ch is not None:
                rAU = true_frame_in[:, rAU_ch:rAU_ch + 1]
                invDx2 = true_frame_in[:, invDx2_ch:invDx2_ch + 1] if invDx2_ch is not None else None
                invDy2 = true_frame_in[:, invDy2_ch:invDy2_ch + 1] if invDy2_ch is not None else None
                if rhs_ch is not None and cfg.pde_use_rhs_if_available:
                    rhs_f = true_frame_in[:, rhs_ch:rhs_ch + 1]
                    pde_loss = poisson_rAU_residual_loss(
                        prgh_next_pred,
                        rAU,
                        rhs_f,
                        grid_phys.float(),
                        invDx2=invDx2,
                        invDy2=invDy2,
                        interior_margin=int(getattr(cfg, "poisson_check_interior_margin", 1)),
                        normalize_by_rhs=bool(cfg.pde_normalize_by_rhs),
                        rhs_eps_abs=float(cfg.pde_rhs_eps_abs),
                        rhs_max_weight=float(cfg.pde_rhs_max_weight),
                    )
                else:
                    pde_loss = poisson_rAU_consistency_loss(
                        prgh_next_pred,
                        next_true if dp_target.startswith("prgh") else next_true - rho_next * gh,
                        rAU,
                        grid_phys.float(),
                        invDx2=invDx2,
                        invDy2=invDy2,
                        interior_margin=int(getattr(cfg, "poisson_check_interior_margin", 1)),
                    )
            elif rhs_ch is not None and cfg.pde_use_rhs_if_available:
                pde_loss = poisson_residual_loss(
                    p_next_pred,
                    rho_next,
                    true_frame_in[:, rhs_ch:rhs_ch + 1],
                    grid_phys.float(),
                    interior_margin=int(getattr(cfg, "poisson_check_interior_margin", 1)),
                )
            else:
                pde_loss = poisson_consistency_loss(
                    p_next_pred,
                    next_true if not dp_target.startswith("prgh") else next_true + rho_next * gh,
                    rho_next,
                    grid_phys.float(),
                    interior_margin=int(getattr(cfg, "poisson_check_interior_margin", 1)),
                )

            # -------- optional continuity supervision / monitor (eval)
            # Always compute the raw continuity monitor when tensors are available.
            # Only the weighted contribution to loss_t remains schedule-controlled.
            w_cont = float(cont_w_sched)
            if getattr(batch, "u_true", None) is not None:
                if bool(getattr(cfg, "cont_final_only", True)):
                    do_cont = t == (Tout_use - 1)
                else:
                    do_cont = True

                if do_cont and dp_target.startswith("prgh") and (rAU_ch is not None) and (invDx2_ch is not None) and (invDy2_ch is not None):
                    frame_u = true_frame_in
                    rAU_f = frame_u[:, rAU_ch:rAU_ch + 1].float()
                    invDx2_f = frame_u[:, invDx2_ch:invDx2_ch + 1].float()
                    invDy2_f = frame_u[:, invDy2_ch:invDy2_ch + 1].float()
                    dx, dy = _infer_dx_dy_from_invDx2(invDx2_f, invDy2_f)

                    prgh_true_final = next_true
                    dpdx_true, dpdy_true = _grad2d(prgh_true_final, dx, dy)
                    dpdx_pred, dpdy_pred = _grad2d(prgh_next_pred.float(), dx, dy)

                    U_true = batch.u_true.float()  # (B,2,H,W)
                    Ux_true = U_true[:, 0:1]
                    Uy_true = U_true[:, 1:2]

                    HbyA_x = Ux_true + rAU_f * dpdx_true
                    HbyA_y = Uy_true + rAU_f * dpdy_true
                    Ux_pred = HbyA_x - rAU_f * dpdx_pred
                    Uy_pred = HbyA_y - rAU_f * dpdy_pred
                    div_pred = _div2d(Ux_pred, Uy_pred, dx, dy)

                    margin = getattr(cfg, "cont_interior_margin", None)
                    if margin is None:
                        margin = int(getattr(cfg, "poisson_check_interior_margin", 1))
                    m_int = inner_mask(mask.float(), margin=int(margin))
                    if float(m_int.sum().detach().cpu()) <= 0.0:
                        m_int = mask.float()
                    cont_loss = masked_mse(div_pred, torch.zeros_like(div_pred), m_int)

            # -------- optional U supervision (final-corrector only)
            u_loss = torch.zeros((), device=device)
            w_u = float(getattr(cfg, "u_loss_weight", 0.0))
            if w_u > 0.0 and (batch.u_true is not None):
                do_u = True
                if bool(getattr(cfg, "u_loss_final_only", True)):
                    do_u = (t == Tout_use - 1) and (Tout_use == cfg.T_out)
                if do_u and dp_target.startswith("prgh"):
                    frame_u = (phys_fut[:, -1] if phys_fut.shape[1] > 0 else phys_win[:, -1]).float()
                    rAU_f = frame_u[:,
                            cfg.rAU_channel:cfg.rAU_channel + 1].float() if cfg.rAU_channel is not None else None
                    invDx2_f = frame_u[:,
                               cfg.invDx2_channel:cfg.invDx2_channel + 1].float() if cfg.invDx2_channel is not None else None
                    invDy2_f = frame_u[:,
                               cfg.invDy2_channel:cfg.invDy2_channel + 1].float() if cfg.invDy2_channel is not None else None
                    if (rAU_f is not None) and (invDx2_f is not None) and (invDy2_f is not None) and (
                            prgh_ch is not None):
                        dx, dy = _infer_dx_dy_from_invDx2(invDx2_f, invDy2_f)
                        prgh_true_in_f = frame_u[:, prgh_ch:prgh_ch + 1].float()
                        prgh_true_final = prgh_true_in_f + dp_fut[:, -1].float()
                        dpdx_true, dpdy_true = _grad2d(prgh_true_final, dx, dy)
                        dpdx_pred, dpdy_pred = _grad2d(prgh_next_pred.float(), dx, dy)
                        U_true = batch.u_true.float()
                        Ux_true = U_true[:, 0:1]
                        Uy_true = U_true[:, 1:2]
                        HbyA_x = Ux_true + rAU_f * dpdx_true
                        HbyA_y = Uy_true + rAU_f * dpdy_true
                        Ux_pred = HbyA_x - rAU_f * dpdx_pred
                        Uy_pred = HbyA_y - rAU_f * dpdy_pred
                        U_pred = torch.cat([Ux_pred, Uy_pred], dim=1)
                        Bc, _, H, W = prgh_next_pred.shape
                        u_pred_flat = U_pred.reshape(Bc * 2, 1, H, W)
                        u_true_flat = U_true.reshape(Bc * 2, 1, H, W)
                        mask_flat = mask.repeat_interleave(2, dim=0)
                        u_loss = masked_mse(u_pred_flat, u_true_flat, mask_flat)

                        # total val loss (match training objective)
            w_grad = float(cfg.grad_loss_weight)
            w_pnext = float(cfg.p_next_loss_weight)
            w_pnext_dm = float(getattr(cfg, "p_next_demean_loss_weight", 0.0))
            w_spec = float(cfg.spec_loss_weight)
            w_pde = float(pde_w_sched)
            w_cont = float(cont_w_sched)

            loss_t = (
                    dp_w * dp_loss
                    + w_grad * grad_loss
                    + w_pnext * pnext_loss
                    + w_pnext_dm * pnext_demean_loss
                    + w_spec * spec_loss
                    + float(getattr(cfg, 'gauge_loss_weight', 0.0)) * gauge_loss
            )
            use_logvars = bool(getattr(cfg, "use_learned_logvars", False)) and hasattr(model, "loss_logvars")

            if (w_pde > 0.0):
                if use_logvars and ("pde" in model.loss_logvars):  # type: ignore[attr-defined]
                    s_pde = model.loss_logvars["pde"]  # type: ignore[attr-defined]
                    loss_t = loss_t + w_pde * _logvar_weighted_term(pde_loss, s_pde)
                else:
                    loss_t = loss_t + w_pde * pde_loss

            if (w_cont > 0.0):
                if use_logvars and ("cont" in model.loss_logvars):  # type: ignore[attr-defined]
                    s_cont = model.loss_logvars["cont"]  # type: ignore[attr-defined]
                    loss_t = loss_t + w_cont * _logvar_weighted_term(cont_loss, s_cont)
                else:
                    loss_t = loss_t + w_cont * cont_loss

            w_u = float(getattr(cfg, "u_loss_weight", 0.0))
            if w_u > 0.0:
                loss_t = loss_t + w_u * u_loss

            step_losses.append(loss_t)
            step_rels.append(rel_t)

            # accum logs
            comp_sum["dp"] += float(dp_loss.detach().cpu())
            comp_sum["grad"] += float(grad_loss.detach().cpu())
            comp_sum["pnext"] += float(pnext_loss.detach().cpu())
            comp_sum["pnext_dm"] += float(pnext_demean_loss.detach().cpu())
            comp_sum["spec"] += float(spec_loss.detach().cpu())
            comp_sum["gauge"] += float(gauge_loss.detach().cpu())
            comp_sum["pde"] += float(pde_loss.detach().cpu())
            comp_sum["cont"] += float(cont_loss.detach().cpu())
            comp_sum["u"] += float(u_loss.detach().cpu())

            comp_w_sum["dp_w"] += float((dp_w * dp_loss).detach().cpu())
            comp_w_sum["grad_w"] += float((w_grad * grad_loss).detach().cpu())
            comp_w_sum["pnext_w"] += float((w_pnext * pnext_loss).detach().cpu())
            comp_w_sum["pnext_dm_w"] += float((w_pnext_dm * pnext_demean_loss).detach().cpu())
            comp_w_sum["spec_w"] += float((w_spec * spec_loss).detach().cpu())
            comp_w_sum["gauge_w"] += float((float(getattr(cfg, "gauge_loss_weight", 0.0)) * gauge_loss).detach().cpu())
            # Weighted contributions: if using learned logvars, log the *actual* added term (w*(exp(-s)*L + s)).
            use_logvars = bool(getattr(cfg, "use_learned_logvars", False)) and hasattr(model, "loss_logvars")

            if (w_pde > 0.0) and use_logvars and ("pde" in model.loss_logvars):  # type: ignore[attr-defined]
                s_pde = model.loss_logvars["pde"]  # type: ignore[attr-defined]
                pde_term = w_pde * _logvar_weighted_term(pde_loss, s_pde)
                comp_w_sum["pde_w"] += float(pde_term.detach().cpu())
                comp_sum["pde_logv"] = comp_sum.get("pde_logv", 0.0) + float(s_pde.detach().cpu())
                comp_sum["pde_w_eff"] = comp_sum.get("pde_w_eff", 0.0) + float((torch.exp(-s_pde) * w_pde).detach().cpu())
            else:
                comp_w_sum["pde_w"] += float((w_pde * pde_loss).detach().cpu())

            if (w_cont > 0.0) and use_logvars and ("cont" in model.loss_logvars):  # type: ignore[attr-defined]
                s_cont = model.loss_logvars["cont"]  # type: ignore[attr-defined]
                cont_term = w_cont * _logvar_weighted_term(cont_loss, s_cont)
                comp_w_sum["cont_w"] += float(cont_term.detach().cpu())
                comp_sum["cont_logv"] = comp_sum.get("cont_logv", 0.0) + float(s_cont.detach().cpu())
                comp_sum["cont_w_eff"] = comp_sum.get("cont_w_eff", 0.0) + float((torch.exp(-s_cont) * w_cont).detach().cpu())
            else:
                comp_w_sum["cont_w"] += float((w_cont * cont_loss).detach().cpu())
            comp_w_sum["u_w"] += float((w_u * u_loss).detach().cpu())

            n_terms += 1

            # pure rollout in eval (no TF): only shift if we have a next frame
            if has_next:
                next_frame = phys_fut[:, t].clone()
                if dp_target.startswith("prgh"):
                    next_frame[:, prgh_ch:prgh_ch + 1] = prgh_next_pred
                    _overwrite_p_if_needed(
                        cfg,
                        next_frame=next_frame,
                        prgh_for_p=prgh_next_pred,
                        grid=grid_phys,
                        alpha_ch=alpha_ch,
                        rho_ch=rho_ch,
                        invrho_ch=invrho_ch,
                        p_ch=p_ch,
                    )
                else:
                    next_frame[:, p_ch:p_ch + 1] = p_next_pred
                    _overwrite_prgh_if_needed(
                        cfg,
                        next_frame=next_frame,
                        p_for_prgh=p_next_pred,
                        grid=grid_phys,
                        alpha_ch=alpha_ch,
                        rho_ch=rho_ch,
                        invrho_ch=invrho_ch,
                        prgh_ch=prgh_ch,
                    )
                phys_win = torch.cat([phys_win[:, 1:], next_frame.unsqueeze(1)], dim=1)



        # Rollout-end diagnostics (align with inference: final p_rgh after rollout)
        rollout_rel = 0.0
        rollout_mae = 0.0
        rollout_pde = 0.0
        if prgh_ch is not None:
            prgh_pred_end = phys_win[:, -1, prgh_ch:prgh_ch + 1].float()

            frame_end = None
            prgh_true_end = None

            if str(getattr(cfg, "dataset_mode", "time")).strip().lower() == "corrector":
                # Corrector mode: the final truth is reconstructed from the last available input frame
                # plus the last delta target. phys_fut has length T_out-1, so direct indexing by Tout_use fails.
                if phys_fut.shape[1] > 0:
                    frame_end = phys_fut[:, min(max(Tout_use - 2, 0), phys_fut.shape[1] - 1)].float()
                else:
                    frame_end = batch.phys_hist[:, -1].float()
                prgh_true_in_end = frame_end[:, prgh_ch:prgh_ch + 1].float()
                prgh_true_end = prgh_true_in_end + dp_fut[:, min(Tout_use - 1, dp_fut.shape[1] - 1)].float()
            elif phys_fut.shape[1] >= Tout_use:
                frame_end = phys_fut[:, Tout_use - 1].float()
                prgh_true_end = frame_end[:, prgh_ch:prgh_ch + 1].float()

            if prgh_true_end is not None and frame_end is not None:
                rollout_rel = float(masked_rel_l2(prgh_pred_end, prgh_true_end, mask).detach().cpu())
                rollout_mae = float(masked_mae(prgh_pred_end, prgh_true_end, mask).mean().detach().cpu())

                if (rhs_ch is not None) and (rAU_ch is not None):
                    rhs_end = frame_end[:, rhs_ch:rhs_ch + 1].float()
                    rAU_end = frame_end[:, rAU_ch:rAU_ch + 1].float()
                    invDx2_end = (frame_end[:, invDx2_ch:invDx2_ch + 1].float() if invDx2_ch is not None else None)
                    invDy2_end = (frame_end[:, invDy2_ch:invDy2_ch + 1].float() if invDy2_ch is not None else None)
                    rollout_pde = float(
                        poisson_rAU_residual_loss(
                            prgh_pred_end, rAU_end, rhs_end, batch.grid_phys,
                            invDx2=invDx2_end, invDy2=invDy2_end,
                            interior_margin=int((cfg.pde_interior_margin if getattr(cfg, 'pde_interior_margin', None) is not None else getattr(cfg, 'poisson_check_interior_margin', 1))),
                            normalize_by_rhs=bool(cfg.pde_normalize_by_rhs),
                            rhs_eps_abs=float(cfg.pde_rhs_eps_abs),
                            rhs_max_weight=float(cfg.pde_rhs_max_weight),
                        ).detach().cpu()
                    )

        comp_sum["rollout_rel"] += float(rollout_rel)
        comp_sum["rollout_mae"] += float(rollout_mae)
        comp_sum["rollout_pde"] += float(rollout_pde)
        loss = torch.stack(step_losses).mean()
        rel = torch.stack(step_rels).mean()

        loss_sum += float(loss.cpu())
        rel_sum += float(rel.cpu())

    denom = max(n_terms, 1)
    comp_avg = {k: v / denom for k, v in comp_sum.items()}
    comp_w_avg = {k: v / denom for k, v in comp_w_sum.items()}

    return loss_sum / max(n_batches, 1), rel_sum / max(n_batches, 1), {**comp_avg, **comp_w_avg}


# =========================
# Main entry
# =========================

def run_training(cfg: TrainConfig) -> None:
    cfg.out_dir = Path(cfg.out_dir)
    cfg.patch_root = Path(cfg.patch_root)
    if cfg.u_patch_root is not None:
        cfg.u_patch_root = Path(cfg.u_patch_root)
    if cfg.resume_path is not None:
        cfg.resume_path = Path(cfg.resume_path)

    validate_config(cfg)
    ensure_dir(cfg.out_dir)
    seed_everything(cfg.seed)

    device = get_device()
    print(f"[info] device={device}")
    runtime_info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device": str(device),
        "seed": cfg.seed,
        "num_workers": cfg.num_workers,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor()
        ),
    }
    save_json(runtime_info, cfg.out_dir / "runtime_info.json")

    # list and split cases
    case_dirs = list_case_dirs(cfg.patch_root)
    if len(case_dirs) == 0:
        raise RuntimeError(f"No cases found under patch_root={cfg.patch_root}")

    train_cases, val_cases, test_cases = split_cases_by_ids(
        case_dirs,
        cfg.train_case_ids,
        cfg.val_case_ids,
        cfg.test_case_ids,
    )
    split_payload = {
        "train": [p.name for p in train_cases],
        "validation": [p.name for p in val_cases],
        "test": [p.name for p in test_cases],
    }
    split_path = cfg.out_dir / "dataset_split.json"
    had_split_manifest = split_path.exists()
    norm_path = cfg.out_dir / "norm_stats.json"
    if norm_path.exists() and not had_split_manifest:
        raise RuntimeError(
            f"Found {norm_path} without dataset_split.json; cannot verify which cases produced it. "
            "Use a new output directory or remove the stale normalization file."
        )
    if had_split_manifest:
        previous_split = load_json(split_path)
        if previous_split != split_payload:
            raise RuntimeError(
                f"Existing dataset split in {split_path} does not match this run. "
                "Use a new output directory or restore the original dataset."
            )
    else:
        save_json(split_payload, split_path)
    print(
        "[info] paper split: "
        f"train={split_payload['train']}, validation={split_payload['validation']}, "
        f"test={split_payload['test']}"
    )

    # compute / load norm stats
    if norm_path.exists():
        norm = NormStats.from_json(load_json(norm_path))
        print(f"[info] loaded norm stats from {norm_path}")
    else:
        print("[info] computing norm stats...")
        norm = compute_norm_stats(
            train_cases,
            max_files_per_case=cfg.max_norm_files_per_case,
            sample_patches_per_file=cfg.sample_patches_per_file,
            seed=cfg.seed,
        )
        save_json(norm.to_json(), norm_path)
        print(f"[info] saved norm stats to {norm_path}")

    # dataset class selection
    mode = (cfg.dataset_mode or "time").strip().lower()
    if mode not in {"time", "corrector"}:
        raise ValueError(f"dataset_mode must be 'time' or 'corrector', got: {cfg.dataset_mode!r}")

    ds_cls = PatchCorrectorWindowIterable if mode == "corrector" else PatchTimeWindowIterable

    def _make_ds(cases: List[Path], *, batch_size: int, steps_per_epoch: int, seed: int):
        kwargs = dict(
            T_in=cfg.T_in,
            T_out=cfg.T_out,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            seed=seed,
            cache_size=cfg.cache_size,
        )
        if ds_cls is PatchCorrectorWindowIterable:
            kwargs["u_patch_root"] = cfg.u_patch_root
        return ds_cls(cases, **kwargs)

    # build a tiny batch to infer n_phys -> set model in_channels
    probe_ds = _make_ds(train_cases, batch_size=min(cfg.batch_size, 2), steps_per_epoch=1, seed=cfg.seed)
    probe = next(iter(probe_ds))
    n_phys = int(probe.phys_hist.shape[2])
    in_channels = n_phys * cfg.T_in + 3
    patch_shape = tuple(int(v) for v in probe.phys_hist.shape[-2:])
    if n_phys != 7 or in_channels != 10:
        raise ValueError(
            "The paper model requires seven physical channels and ten total input channels; "
            f"got n_phys={n_phys}, in_channels={in_channels}."
        )
    if patch_shape != (64, 64):
        raise ValueError(
            f"The reported training uses 64x64 patches; found {patch_shape}."
        )
    if probe.u_true is None:
        raise ValueError("Velocity patches are required for the reported continuity loss")

    # model
    mcfg = FNOConfig(
        in_channels=in_channels,
        out_channels=1,
        modes=cfg.modes,
        width=cfg.width,
        padding=cfg.padding,
        use_residual=cfg.use_residual,
        residual_unet_depth=cfg.residual_unet_depth,
        residual_unet_base_channels=cfg.residual_unet_base_channels,
        residual_unet_gn_groups=cfg.residual_unet_gn_groups,
        residual_unet_dropout=cfg.residual_unet_dropout,
        noise_std=cfg.noise_std
    )
    model = FNO2dEnhanced(mcfg).to(device)
    # Optional: learned log-variance weighting (PDE + continuity only)
    if bool(getattr(cfg, "use_learned_logvars", False)):
        _attach_learned_logvars(model, cfg, device)

    print(f"[info] model params: {count_params(model):,}  in_channels={in_channels}")

    # loaders
    train_ds = _make_ds(train_cases, batch_size=cfg.batch_size, steps_per_epoch=cfg.steps_per_epoch, seed=cfg.seed)
    val_ds = _make_ds(
        val_cases,
        batch_size=cfg.batch_size,
        steps_per_epoch=max(50, cfg.steps_per_epoch // 5),
        seed=cfg.seed + 123
    )

    train_loader = DataLoader(train_ds, batch_size=None, num_workers=cfg.num_workers,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=None, num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))

    # --- Poisson alignment sanity check (optional)
    if cfg.do_poisson_alignment_check and (cfg.rhs_channel is not None) and (cfg.rAU_channel is not None):
        chk_ds = _make_ds(train_cases, batch_size=cfg.batch_size, steps_per_epoch=int(cfg.poisson_check_batches),
                          seed=cfg.seed + 999)
        chk_ds.set_epoch(0)
        chk_loader = DataLoader(chk_ds, batch_size=None, num_workers=0)
        stats = poisson_alignment_check(chk_loader, cfg=cfg, device=device)
        print(
            f"[sanity] Poisson alignment (true p_rgh vs rhs/rAU): rel_rmse={stats.get('rel_rmse', float('nan')):.4g}, mae/rhs_mae={stats.get('mae_over_rhs_mae', float('nan')):.4g}, mse={stats.get('mse', float('nan')):.4g}, samples={int(stats.get('samples', 0))}{(', steps=%s' % int(stats.get('steps', 0))) if int(stats.get('steps', 0))>0 else ''}{(', margin=%s' % int(stats.get('interior_margin', 0))) if 'interior_margin' in stats else ''}{(', best_plus=%.2f' % float(stats.get('best_plus_frac', 0.0))) if 'best_plus_frac' in stats else ''}")
        save_json(stats, cfg.out_dir / 'poisson_alignment.json')

    # --- Continuity scale sanity check (optional)
    if getattr(cfg, 'do_continuity_sanity_check', True) and (cfg.u_patch_root is not None):
        chk_ds2 = _make_ds(train_cases, batch_size=cfg.batch_size, steps_per_epoch=int(getattr(cfg, 'cont_check_batches', 32)),
                           seed=cfg.seed + 1001)
        chk_ds2.set_epoch(0)
        chk_loader2 = DataLoader(chk_ds2, batch_size=None, num_workers=0)
        cstats = continuity_sanity_check(chk_loader2, cfg=cfg, device=device)
        if int(cstats.get('enabled', 0)) == 1:
            print(f"[sanity] Continuity scale (div(U_true), 1/s): rms={cstats.get('div_rms', float('nan')):.4g}, mae={cstats.get('div_mae', float('nan')):.4g}, dx≈{cstats.get('dx_mean', float('nan')):.4g}, dy≈{cstats.get('dy_mean', float('nan')):.4g}, samples={int(cstats.get('samples', 0))}, margin={int(cstats.get('interior_margin', 0))}")
        save_json(cstats, cfg.out_dir / 'continuity_sanity.json')

    # optim
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=(cfg.amp and device.type == "cuda"))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    best_val = float('inf')
    best_rel = float('inf')  # rollout-end relL2 (lower is better)

    # LR schedule state (step-based)
    state: Dict[str, Any] = {
        'global_step': 0,
        'total_steps': int(cfg.epochs * cfg.steps_per_epoch),
        'warmup_steps': int(cfg.warmup_epochs * cfg.steps_per_epoch),
    }

    # checkpoint resume
    start_epoch = 0
    if cfg.resume:
        ckpt_path = cfg.resume_path if cfg.resume_path is not None else (cfg.out_dir / 'ckpt_last.pt')
        if ckpt_path.exists():
            print(f"[info] resuming from {ckpt_path}")
            resume_keys = (
                "dataset_mode", "T_in", "T_out", "modes", "width", "padding",
                "use_residual", "residual_unet_depth", "residual_unet_base_channels",
                "residual_unet_gn_groups", "residual_unet_dropout", "epochs",
                "batch_size", "steps_per_epoch", "lr", "weight_decay", "warmup_epochs",
                "tf_start", "tf_end", "tf_decay_portion", "tf_step_gamma", "tf_step_end",
                "curriculum_tout1", "curriculum_epoch1", "curriculum_tout2",
                "curriculum_epoch2", "offset_aug_max", "offset_aug_prob",
                "dp_loss_weight", "grad_loss_weight", "p_next_demean_loss_weight",
                "spec_loss_weight", "pde_loss_weight", "cont_loss_weight",
                "pde_warmup_epochs", "pde_ramp_epochs", "cont_warmup_epochs",
                "cont_ramp_epochs", "pde_rhs_eps_abs", "pde_rhs_max_weight",
                "normalize_channels", "train_case_ids", "val_case_ids", "test_case_ids",
                "seed", "num_workers",
            )
            expected_resume_cfg = {key: getattr(cfg, key) for key in resume_keys}
            ckpt = _try_load_checkpoint(
                ckpt_path,
                model=model,
                opt=opt,
                scaler=scaler,
                device=device,
                expected_cfg=expected_resume_cfg,
            )
            best_val = float(ckpt.get('best_val', best_val))
            best_rel = float(ckpt.get('best_rel', best_rel))
            start_epoch = int(ckpt.get('epoch', -1)) + 1
            state['global_step'] = int(ckpt.get('global_step', start_epoch * cfg.steps_per_epoch))
            # safety: avoid weird schedule if someone changed epochs/steps
            state['total_steps'] = max(int(state['total_steps']), int(state['global_step']) + 1)
            # Reset optional learned log variances after resume.
            if getattr(cfg, "use_learned_logvars", False) and hasattr(model, "loss_logvars"):
                init = float(getattr(cfg, "learned_logvar_init", 0.0))
                tasks = set(getattr(cfg, "learned_logvar_tasks", ()))
                for name, p in model.loss_logvars.items():  # type: ignore[attr-defined]
                    if (not tasks) or (name in tasks):
                        p.data.fill_(init)
                print(f"[info] reset loss_logvars {sorted(list(tasks)) if tasks else 'ALL'} to init={init}")

    # save config
    save_json({k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()},
              cfg.out_dir / "train_config.json")

    def _fmt_terms(d: Dict[str, float]) -> str:
        # raw terms + weighted terms; print only keys that exist
        if not d:
            return ""
        keys = [
            "dp", "grad", "pnext", "pnext_dm", "spec", "gauge", "pde", "cont", "u",
            "rollout_rel", "rollout_mae", "rollout_pde",
            "pde_logv", "cont_logv", "pde_w_eff", "cont_w_eff",
            "dp_w", "grad_w", "pnext_w", "pnext_dm_w", "spec_w", "gauge_w", "pde_w", "cont_w", "u_w",
        ]
        parts = []
        for k in keys:
            if k in d:
                parts.append(f"{k}={d[k]:.3e}")
        return "  ".join(parts)

    def _flatten_epoch_row(
            epoch: int,
            global_step: int,
            lr_now: float,
            tf_ratio: float,
            tout_use: int,
            pde_w: float,
            train_loss: float,
            val_loss: float,
            val_rel: float,
            tr: Dict[str, float],
            vr: Dict[str, float],
            best_val: float,
            best_rel: float,
    ) -> Dict[str, float]:
        row: Dict[str, float] = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "lr": float(lr_now),
            "tf_ratio": float(tf_ratio),
            "tout_use": int(tout_use),
            "pde_w_sched": float(pde_w),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_relL2": float(val_rel),
            "best_val": float(best_val),
            "best_rollout_rel": float(best_rel),
        }

        for k, v in (tr or {}).items():
            row[f"train_{k}"] = float(v)
        for k, v in (vr or {}).items():
            row[f"val_{k}"] = float(v)

        return row

    def _append_row_to_csv(csv_path: Path, row: Dict[str, float]) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        file_exists = csv_path.exists()
        old_rows: List[Dict[str, str]] = []

        if file_exists:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                old_rows = list(reader)
                old_fields = reader.fieldnames or []
        else:
            old_fields = []

        new_fields = list(row.keys())
        all_fields = list(dict.fromkeys(list(old_fields) + new_fields))

        need_rewrite = (not file_exists) or (all_fields != list(old_fields))

        if need_rewrite:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_fields)
                writer.writeheader()
                for r in old_rows:
                    writer.writerow(r)
                writer.writerow(row)
        else:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_fields)
                writer.writerow(row)

    for epoch in range(start_epoch, cfg.epochs):
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        if hasattr(val_ds, "set_epoch"):
            val_ds.set_epoch(epoch)

        # train/eval now return (total_loss, terms_dict)
        train_loss, tr = train_one_epoch(
            model, train_loader, opt, scaler, device,
            cfg=cfg, norm=norm, epoch=epoch, state=state
        )
        val_loss, val_rel, vr = eval_one_epoch(
            model, val_loader, device,
            cfg=cfg, norm=norm, epoch=epoch
        )

        pde_w = _pde_weight(cfg, epoch)
        tf_ratio = _teacher_forcing_ratio(cfg, epoch)
        lr_now = float(opt.param_groups[0]['lr'])
        tout_use = _tout_for_epoch(cfg, epoch)

        print(
            f"[epoch {epoch:03d}] step={state['global_step']:07d} lr={lr_now:.3e} "
            f"train={train_loss:.6f}  val={val_loss:.6f}  relL2={val_rel:.6f}  rollRel={vr.get('rollout_rel', float('nan')):.6f}  rollPDE={vr.get('rollout_pde', float('nan')):.3e} "
            f"tf={tf_ratio:.3f}  tout={tout_use}  pde_w={pde_w:.4g}"
        )
        if tr:
            print(f"           train_terms: {_fmt_terms(tr)}")
        if vr:
            print(f"           val_terms  : {_fmt_terms(vr)}")

        # save last (Python)
        torch.save(model.state_dict(), cfg.out_dir / "model_last.pt")
        # optional: export portable CPU state_dict for C++ libtorch
        if cfg.export_libtorch_state_dict and (not cfg.export_libtorch_best_only):
            stem = Path(cfg.export_libtorch_name).stem
            suf = Path(cfg.export_libtorch_name).suffix or ".pth"
            _export_libtorch_state_dict(cfg.out_dir / f"{stem}_last{suf}", model=model)

        if cfg.save_ckpt:
            _save_checkpoint(cfg.out_dir / 'ckpt_last.pt', model=model, opt=opt, scaler=scaler, epoch=epoch,
                             global_step=int(state['global_step']), best_val=best_val, best_rel=best_rel, cfg=cfg)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), cfg.out_dir / "model_best.pt")
            if cfg.export_libtorch_state_dict:
                _export_libtorch_state_dict(cfg.out_dir / cfg.export_libtorch_name, model=model)
            if cfg.save_ckpt:
                _save_checkpoint(cfg.out_dir / 'ckpt_best.pt', model=model, opt=opt, scaler=scaler, epoch=epoch,
                                 global_step=int(state['global_step']), best_val=best_val, best_rel=best_rel, cfg=cfg)

        # best-by-rollout-rel checkpoint (align with inference goal)
        val_roll_rel = float(vr.get('rollout_rel', val_rel)) if vr else float(val_rel)
        if val_roll_rel < best_rel:
            best_rel = val_roll_rel
            torch.save(model.state_dict(), cfg.out_dir / 'model_best_rel.pt')
            if cfg.save_ckpt:
                _save_checkpoint(cfg.out_dir / 'ckpt_best_rel.pt', model=model, opt=opt, scaler=scaler, epoch=epoch,
                                 global_step=int(state['global_step']), best_val=best_val, best_rel=best_rel, cfg=cfg)

        # save epoch logs to csv (after best metrics are updated)
        row = _flatten_epoch_row(
            epoch=epoch,
            global_step=int(state["global_step"]),
            lr_now=lr_now,
            tf_ratio=tf_ratio,
            tout_use=tout_use,
            pde_w=pde_w,
            train_loss=train_loss,
            val_loss=val_loss,
            val_rel=val_rel,
            tr=tr,
            vr=vr,
            best_val=best_val,
            best_rel=best_rel,
        )
        _append_row_to_csv(cfg.out_dir / "train_log.csv", row)
