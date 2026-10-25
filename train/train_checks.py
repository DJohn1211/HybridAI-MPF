"""
Sanity / diagnostic checks used by the dp_rgh / Δp_rgh trainer.

Kept separate so the main trainer stays readable.
"""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_timewindow import Batch

from losses import get_rho_from_alpha, inner_mask

from train_helpers import (
    _compute_gh,
    _resolve_channel,
    _rho_from_frame,
    _infer_dx_dy_from_invDx2,
    _div2d,
)

if TYPE_CHECKING:
    from trainer_timewindow_dp_rgh import TrainConfig


@torch.no_grad()
def poisson_alignment_check(
        loader: DataLoader,
        *,
        cfg: TrainConfig,
        device: torch.device,
) -> Dict[str, float]:
    """
    Pre-training Poisson consistency check.

    This diagnostic complements validate_pressure_dataset.py for corrector data.

    Corrector-mode semantics (ONE physical time step, k=0..K-1):
      - Use step-k inputs: (rhs_k, rAU_k, p_rgh_in_k, invDx2, invDy2)
      - Reconstruct step-k post-solve pressure:
            p_rgh_out_k = p_rgh_in_k + Δp_rgh_true_k
        (or if dp_target='p': convert to p_rgh_out via rho*gh)
      - Check discrete residual with SAME rhs_k/rAU_k:
            div(rAU_k * grad(p_rgh_out_k)) ?= rhs_k

    NOTE on patches:
      - The discrete operator needs boundary conditions. A patch boundary is usually
        *not* a physical domain boundary, so the boundary cells produce artificial residual.
      - We therefore exclude a small interior margin (default: 1 cell) from metrics.

    In time-mode, we keep the previous behavior.
    """

    # require rhs + rAU
    if cfg.rhs_channel is None or cfg.rAU_channel is None:
        return {"enabled": 0.0}

    mode = (getattr(cfg, "dataset_mode", "time") or "time").strip().lower()

    mse_sum = 0.0
    relrmse_sum = 0.0
    ratio_mae_sum = 0.0
    best_plus_frac = 0.0
    n = 0

    interior_margin = int(getattr(cfg, "poisson_check_interior_margin", 1))
    interior_margin = max(interior_margin, 0)

    def _apply_div_rAU_grad_like_sanity(
            p: torch.Tensor,
            rAU: torch.Tensor,
            invDx2: torch.Tensor,
            invDy2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Torch version of sanity_check.apply_div_rAU_grad() on structured grid.

        p,rAU,invDx2,invDy2: (B,1,H,W) float32
        returns: (B,1,H,W) float32
        """
        B, _, H, W = p.shape
        out = torch.zeros_like(p)

        # x-direction
        if W >= 2:
            rAUf_x = 0.5 * (rAU[:, :, :, 1:] + rAU[:, :, :, :-1])    # (B,1,H,W-1)
            dp_x = (p[:, :, :, 1:] - p[:, :, :, :-1])                # (B,1,H,W-1)
            flux_x = rAUf_x * dp_x

            if W > 2:
                out[:, :, :, 1:-1] = out[:, :, :, 1:-1] + (flux_x[:, :, :, 1:] - flux_x[:, :, :, :-1]) * invDx2[:, :, :, 1:-1]
            out[:, :, :, 0] = out[:, :, :, 0] + (flux_x[:, :, :, 0]) * invDx2[:, :, :, 0]
            out[:, :, :, -1] = out[:, :, :, -1] + (-flux_x[:, :, :, -1]) * invDx2[:, :, :, -1]

        # y-direction
        if H >= 2:
            rAUf_y = 0.5 * (rAU[:, :, 1:, :] + rAU[:, :, :-1, :])    # (B,1,H-1,W)
            dp_y = (p[:, :, 1:, :] - p[:, :, :-1, :])                # (B,1,H-1,W)
            flux_y = rAUf_y * dp_y

            if H > 2:
                out[:, :, 1:-1, :] = out[:, :, 1:-1, :] + (flux_y[:, :, 1:, :] - flux_y[:, :, :-1, :]) * invDy2[:, :, 1:-1, :]
            out[:, :, 0, :] = out[:, :, 0, :] + (flux_y[:, :, 0, :]) * invDy2[:, :, 0, :]
            out[:, :, -1, :] = out[:, :, -1, :] + (-flux_y[:, :, -1, :]) * invDy2[:, :, -1, :]

        return out




    for bi, batch in enumerate(loader):
        if bi >= int(cfg.poisson_check_batches):
            break

        batch = Batch(
            phys_hist=batch.phys_hist.to(device),
            phys_fut=batch.phys_fut.to(device),
            dp_fut=batch.dp_fut.to(device),
            grid=batch.grid.to(device),
            grid_phys=batch.grid_phys.to(device),
            u_true=(batch.u_true.to(device) if (getattr(batch, "u_true", None) is not None) else None),
        )

        B, Tin, n_phys, H, W = batch.phys_hist.shape
        alpha_ch = _resolve_channel(cfg.alpha_channel, n_phys)
        p_ch = _resolve_channel(cfg.p_channel, n_phys)
        rhs_ch = _resolve_channel(cfg.rhs_channel, n_phys)
        rAU_ch = _resolve_channel(cfg.rAU_channel, n_phys)
        prgh_ch = _resolve_channel(cfg.prgh_channel, n_phys) if cfg.prgh_channel is not None else None
        invDx2_ch = _resolve_channel(cfg.invDx2_channel, n_phys) if cfg.invDx2_channel is not None else None
        invDy2_ch = _resolve_channel(cfg.invDy2_channel, n_phys) if cfg.invDy2_channel is not None else None
        invrho_ch = _resolve_channel(cfg.invrho_channel, n_phys) if cfg.invrho_channel is not None else None
        rho_ch = _resolve_channel(cfg.rho_channel, n_phys) if cfg.rho_channel is not None else None

        # mask from grid (third channel)
        m = batch.grid[:, 2:3].float()

        dp_target = str(getattr(cfg, "dp_target", "prgh")).lower()
        gh = _compute_gh(cfg, batch.grid_phys.float())

        # -------- corrector-mode (sanity_check-consistent) --------
        if mode == "corrector":
            K = int(cfg.T_out)
            if K <= 0:
                continue

            max_steps = int(getattr(cfg, "poisson_check_max_steps", 1))
            if max_steps <= 0:
                continue

            # Evaluate LAST steps by default (more representative); e.g. max_steps=1 -> only corr(K-1)
            if max_steps >= K:
                steps = list(range(K))
            else:
                steps = list(range(K - max_steps, K))

            for k in steps:
                # step-k input frame: k=0 from phys_hist[-1], k>=1 from phys_fut[k-1]
                if k == 0:
                    frame_in = batch.phys_hist[:, -1].float()
                else:
                    kk = k - 1
                    if kk < 0 or kk >= batch.phys_fut.shape[1]:
                        continue
                    frame_in = batch.phys_fut[:, kk].float()

                rhs = frame_in[:, rhs_ch:rhs_ch + 1]
                rAU = frame_in[:, rAU_ch:rAU_ch + 1]

                if invDx2_ch is None or invDy2_ch is None:
                    # Need spacing to build a meaningful operator
                    continue
                invDx2 = frame_in[:, invDx2_ch:invDx2_ch + 1]
                invDy2 = frame_in[:, invDy2_ch:invDy2_ch + 1]

                # reconstruct post-solve pressure for this corrector step
                dp_true = batch.dp_fut[:, k].float()  # (B,1,H,W)

                if dp_target.startswith("prgh"):
                    if prgh_ch is None:
                        rho = _rho_from_frame(cfg, frame=frame_in, alpha_ch=alpha_ch, rho_ch=rho_ch, invrho_ch=invrho_ch)
                        p_in = frame_in[:, p_ch:p_ch + 1]
                        pr_in = p_in - rho * gh
                    else:
                        pr_in = frame_in[:, prgh_ch:prgh_ch + 1]
                    pr_out = pr_in + dp_true
                else:
                    p_in = frame_in[:, p_ch:p_ch + 1]
                    p_out = p_in + dp_true
                    rho = _rho_from_frame(cfg, frame=frame_in, alpha_ch=alpha_ch, rho_ch=rho_ch, invrho_ch=invrho_ch)
                    pr_out = p_out - rho * gh

                # A = div(rAU grad p)
                A = _apply_div_rAU_grad_like_sanity(pr_out, rAU, invDx2, invDy2)

                # crop margin
                if interior_margin > 0:
                    m_use = m.clone()
                    m_use[:, :, :interior_margin, :] = 0.0
                    m_use[:, :, -interior_margin:, :] = 0.0
                    m_use[:, :, :, :interior_margin] = 0.0
                    m_use[:, :, :, -interior_margin:] = 0.0
                else:
                    m_use = m

                den = m_use.sum().clamp_min(1e-12)

                rhs_mse = float(((rhs * rhs) * m_use).sum().cpu() / den.cpu())
                rhs_mae = float(((rhs.abs()) * m_use).sum().cpu() / den.cpu())

                # minus / plus conventions
                res_minus = A - rhs
                res_plus = A + rhs

                mse_a = float(((res_minus * res_minus) * m_use).sum().cpu() / den.cpu())
                relrmse_a = (mse_a / (rhs_mse + 1e-12)) ** 0.5
                mae_a = float((res_minus.abs() * m_use).sum().cpu() / den.cpu())
                ratio_mae_a = mae_a / (rhs_mae + 1e-12)

                mse_b = float(((res_plus * res_plus) * m_use).sum().cpu() / den.cpu())
                relrmse_b = (mse_b / (rhs_mse + 1e-12)) ** 0.5
                mae_b = float((res_plus.abs() * m_use).sum().cpu() / den.cpu())
                ratio_mae_b = mae_b / (rhs_mae + 1e-12)

                use_plus = (relrmse_b < relrmse_a)
                best_plus_frac += 1.0 if use_plus else 0.0

                mse = mse_b if use_plus else mse_a
                relrmse = relrmse_b if use_plus else relrmse_a
                ratio_mae = ratio_mae_b if use_plus else ratio_mae_a

                mse_sum += mse
                relrmse_sum += relrmse
                ratio_mae_sum += ratio_mae
                n += 1

            continue

        # -------- time-mode (keep previous behavior) --------
        max_steps = min(int(cfg.poisson_check_max_steps), max(0, cfg.T_out - 1))
        for t in range(max_steps):
            phys_next = batch.phys_fut[:, t].float()
            rhs = phys_next[:, rhs_ch:rhs_ch + 1]
            rAU = phys_next[:, rAU_ch:rAU_ch + 1]
            invDx2 = phys_next[:, invDx2_ch:invDx2_ch + 1] if invDx2_ch is not None else None
            invDy2 = phys_next[:, invDy2_ch:invDy2_ch + 1] if invDy2_ch is not None else None

            if invDx2 is None or invDy2 is None:
                continue

            if prgh_ch is not None:
                p_rgh_true = phys_next[:, prgh_ch:prgh_ch + 1]
            else:
                alpha = phys_next[:, alpha_ch:alpha_ch + 1]
                rho = get_rho_from_alpha(alpha, rho1=float(cfg.rho1), rho2=float(cfg.rho2))
                p_true = phys_next[:, p_ch:p_ch + 1]
                p_rgh_true = p_true - rho * gh

            A = _apply_div_rAU_grad_like_sanity(p_rgh_true, rAU, invDx2, invDy2)
            res_minus = A - rhs
            res_plus = A + rhs

            # crop margin for time-mode as well (patch boundary)
            if interior_margin > 0:
                m_use = m.clone()
                m_use[:, :, :interior_margin, :] = 0.0
                m_use[:, :, -interior_margin:, :] = 0.0
                m_use[:, :, :, :interior_margin] = 0.0
                m_use[:, :, :, -interior_margin:] = 0.0
            else:
                m_use = m

            den = m_use.sum().clamp_min(1e-12)

            rhs_mse = float(((rhs * rhs) * m_use).sum().cpu() / den.cpu())
            rhs_mae = float(((rhs.abs()) * m_use).sum().cpu() / den.cpu())

            mse_a = float(((res_minus * res_minus) * m_use).sum().cpu() / den.cpu())
            relrmse_a = (mse_a / (rhs_mse + 1e-12)) ** 0.5
            mae_a = float((res_minus.abs() * m_use).sum().cpu() / den.cpu())
            ratio_mae_a = mae_a / (rhs_mae + 1e-12)

            mse_b = float(((res_plus * res_plus) * m_use).sum().cpu() / den.cpu())
            relrmse_b = (mse_b / (rhs_mse + 1e-12)) ** 0.5
            mae_b = float((res_plus.abs() * m_use).sum().cpu() / den.cpu())
            ratio_mae_b = mae_b / (rhs_mae + 1e-12)

            use_plus = (relrmse_b < relrmse_a)
            best_plus_frac += 1.0 if use_plus else 0.0

            mse = mse_b if use_plus else mse_a
            relrmse = relrmse_b if use_plus else relrmse_a
            ratio_mae = ratio_mae_b if use_plus else ratio_mae_a

            mse_sum += mse
            relrmse_sum += relrmse
            ratio_mae_sum += ratio_mae
            n += 1

    if n == 0:
        return {"enabled": 0.0}

    out = {
        "enabled": 1.0,
        "samples": float(n),
        "mse": float(mse_sum / n),
        "rel_rmse": float(relrmse_sum / n),
        "mae_over_rhs_mae": float(ratio_mae_sum / n),
        "best_plus_frac": float(best_plus_frac / max(n, 1)),
        "steps": int(getattr(cfg, "poisson_check_max_steps", 1)),
        "interior_margin": float(interior_margin),
    }
    return out


def continuity_sanity_check(
        loader: DataLoader,
        *,
        cfg: TrainConfig,
        device: torch.device,
) -> Dict[str, float]:
    """Estimate the typical scale of div(U_true) on the interior of patches.

    Units: 1/s. This check helps you choose cont_loss_weight so continuity doesn't overwhelm dp loss.
    Requires u_patch_root (batches must include u_true).
    """
    enabled = bool(getattr(cfg, "do_continuity_sanity_check", True))
    if (not enabled) or (cfg.u_patch_root is None):
        return {"enabled": 0.0}

    margin = getattr(cfg, "cont_interior_margin", None)
    if margin is None:
        margin = int(getattr(cfg, "poisson_check_interior_margin", 1))

    div_vals = []
    dx_vals = []
    dy_vals = []
    n = 0

    for batch in loader:
        batch = Batch(
            phys_hist=batch.phys_hist.to(device),
            phys_fut=batch.phys_fut.to(device),
            dp_fut=batch.dp_fut.to(device),
            grid=batch.grid.to(device),
            grid_phys=batch.grid_phys.to(device),
            u_true=(batch.u_true.to(device) if getattr(batch, "u_true", None) is not None else None),
        )
        if batch.u_true is None:
            continue

        phys_fut = batch.phys_fut
        phys_win = batch.phys_hist
        grid = batch.grid_phys
        mask = grid[:, 2:3].float()

        frame_u = (phys_fut[:, -1] if phys_fut.shape[1] > 0 else phys_win[:, -1]).float()
        invDx2_ch = getattr(cfg, "invDx2_channel", None)
        invDy2_ch = getattr(cfg, "invDy2_channel", None)
        if invDx2_ch is None or invDy2_ch is None:
            continue

        invDx2 = frame_u[:, invDx2_ch:invDx2_ch + 1].float()
        invDy2 = frame_u[:, invDy2_ch:invDy2_ch + 1].float()
        dx, dy = _infer_dx_dy_from_invDx2(invDx2, invDy2)

        U_true = batch.u_true.float()
        div_true = _div2d(U_true[:, 0:1], U_true[:, 1:2], dx, dy)

        m_int = inner_mask(mask, margin=int(margin))
        msum = m_int.sum().clamp_min(1.0)
        div_rms = torch.sqrt(((div_true ** 2) * m_int).sum() / msum)
        div_mae = (div_true.abs() * m_int).sum() / msum

        div_vals.append(torch.stack([div_rms, div_mae]).detach().cpu())
        dx_vals.append(dx.detach().cpu())
        dy_vals.append(dy.detach().cpu())
        n += int(div_true.shape[0])
        if n >= int(getattr(cfg, "cont_check_batches", 32)) * int(getattr(cfg, "batch_size", 1)):
            break

    if len(div_vals) == 0:
        return {"enabled": 0.0}

    div_vals = torch.stack(div_vals, dim=0)
    dx_vals = torch.stack(dx_vals, dim=0)
    dy_vals = torch.stack(dy_vals, dim=0)

    stats = {
        "enabled": 1.0,
        "samples": float(n),
        "div_rms": float(div_vals[:, 0].mean().item()),
        "div_mae": float(div_vals[:, 1].mean().item()),
        "dx_mean": float(dx_vals.mean().item()),
        "dy_mean": float(dy_vals.mean().item()),
        "interior_margin": float(int(margin)),
    }
    return stats
# =========================
# Train / eval steps
# =========================


# =========================
# LR schedule + checkpoint helpers
# =========================
