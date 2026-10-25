from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# =========================
# Basic masked losses
# =========================

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Masked MSE. pred/target/mask: (B,1,H,W)
    NOTE: force float32 math to avoid AMP/FP16 overflow -> NaN.
    """
    pred = pred.float()
    target = target.float()
    mask = mask.float()
    diff2 = (pred - target).pow(2) * mask
    return diff2.sum() / mask.sum().clamp_min(eps)


def masked_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Masked weighted MSE. weight: (B,1,H,W)
    NOTE: force float32 math to avoid AMP/FP16 overflow -> NaN.
    """
    pred = pred.float()
    target = target.float()
    mask = mask.float()
    weight = weight.float()

    w = weight * mask
    diff2 = (pred - target).pow(2) * w
    return diff2.sum() / w.sum().clamp_min(eps)


def masked_rel_l2(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Masked relative L2 on physical or normalized tensors. pred/target/mask: (B,1,H,W)
    NOTE: force float32 math to avoid AMP/FP16 overflow -> NaN.
    """
    pred = pred.float()
    target = target.float()
    mask = mask.float()

    num = ((pred - target).pow(2) * mask).sum()
    den = (target.pow(2) * mask).sum().clamp_min(eps)
    return torch.sqrt(num / den)


# =========================
# Spectral / frequency-domain loss
# =========================

def spectral_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """MSE in frequency domain.

    pred/target: (B,1,H,W)
    If mask is provided, we multiply both fields by mask before FFT.

    Note: We use rfft2 for real inputs; loss is mean(|F(pred)-F(target)|^2).
    """
    pred = pred.float()
    target = target.float()
    if mask is not None:
        m = mask.float()
        pred = pred * m
        target = target * m

    Fp = torch.fft.rfft2(pred, dim=(-2, -1), norm='backward')
    Ft = torch.fft.rfft2(target, dim=(-2, -1), norm='backward')
    diff = Fp - Ft
    # |a+ib|^2 = a^2 + b^2
    loss = (diff.real * diff.real + diff.imag * diff.imag).mean()
    return loss.clamp_min(eps)



# =========================
# Gradient utilities (Sobel)
# =========================

def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
        device=device, dtype=dtype
    ) / 8.0
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0],
         [ 0.0,  0.0,  0.0],
         [ 1.0,  2.0,  1.0]],
        device=device, dtype=dtype
    ) / 8.0
    return kx.view(1, 1, 3, 3), ky.view(1, 1, 3, 3)


def sobel_grad_xy(field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """field: (B,1,H,W) -> gx, gy: (B,1,H,W)"""
    kx, ky = _sobel_kernels(field.device, field.dtype)
    gx = F.conv2d(field, kx, padding=1)
    gy = F.conv2d(field, ky, padding=1)
    return gx, gy


def sobel_grad_mag(field: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    gx, gy = sobel_grad_xy(field)
    return torch.sqrt(gx * gx + gy * gy + eps)


def interface_weight_from_alpha(
    alpha: torch.Tensor,
    *,
    interface_lambda: float = 2.0,
    eps: float = 1e-6,
    clamp_max: float = 10.0,
) -> torch.Tensor:
    """
    alpha: (B,1,H,W)
    Build weight map emphasizing interface via |∇alpha|.
    weight = 1 + lambda * clamp( normalized(|∇alpha|), max=clamp_max )

    NOTE:
      - force float32 math (avoid AMP/FP16 overflow)
      - clamp to prevent rare patches with tiny mean(|∇alpha|) from exploding weights.
    """
    alpha = alpha.float()
    g = sobel_grad_mag(alpha, eps=eps)
    den = g.mean(dim=(2, 3), keepdim=True).clamp_min(eps)
    g_norm = (g / den).clamp(max=float(clamp_max))
    return 1.0 + float(interface_lambda) * g_norm


def masked_grad_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Gradient MSE on physical fields, using Sobel gradients.
    NOTE: force float32 math to avoid AMP/FP16 overflow -> NaN.
    """
    pred = pred.float()
    target = target.float()
    mask = mask.float()
    if weight is not None:
        weight = weight.float()
    pgx, pgy = sobel_grad_xy(pred)
    tgx, tgy = sobel_grad_xy(target)
    if weight is None:
        return masked_mse(pgx, tgx, mask, eps=eps) + masked_mse(pgy, tgy, mask, eps=eps)
    return masked_weighted_mse(pgx, tgx, mask, weight, eps=eps) + masked_weighted_mse(pgy, tgy, mask, weight, eps=eps)


# =========================
# Poisson / PDE residual losses (central difference)
# =========================

def _central_diff_kernels(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    # d/dx with central difference on a 3x3 stencil; same for d/dy
    kx = torch.tensor(
        [[0.0, 0.0, 0.0],
         [-0.5, 0.0, 0.5],
         [0.0, 0.0, 0.0]],
        device=device, dtype=dtype
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[0.0, -0.5, 0.0],
         [0.0,  0.0, 0.0],
         [0.0,  0.5, 0.0]],
        device=device, dtype=dtype
    ).view(1, 1, 3, 3)
    return kx, ky


def _estimate_dxdy_from_xy(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x,y,mask: (B,1,H,W). Estimate dx,dy per batch from coordinate channels.
    Works best on structured grids where x increases along W and y increases along H.
    """
    # dx from x differences along W
    dx_map = (x[..., 1:] - x[..., :-1]).abs()
    mx = mask[..., 1:] * mask[..., :-1]
    dx = (dx_map * mx).sum(dim=(2, 3)) / mx.sum(dim=(2, 3)).clamp_min(eps)  # (B,1)

    # dy from y differences along H
    dy_map = (y[..., 1:, :] - y[..., :-1, :]).abs()
    my = mask[..., 1:, :] * mask[..., :-1, :]
    dy = (dy_map * my).sum(dim=(2, 3)) / my.sum(dim=(2, 3)).clamp_min(eps)  # (B,1)

    return dx.view(-1, 1, 1, 1).clamp_min(eps), dy.view(-1, 1, 1, 1).clamp_min(eps)


def _grad_xy_cd(field: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    kx, ky = _central_diff_kernels(field.device, field.dtype)
    f = F.pad(field, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(f, kx) / dx
    gy = F.conv2d(f, ky) / dy
    return gx, gy


def _div_xy_cd(fx: torch.Tensor, fy: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    kx, ky = _central_diff_kernels(fx.device, fx.dtype)
    fxp = F.pad(fx, (1, 1, 1, 1), mode="replicate")
    fyp = F.pad(fy, (1, 1, 1, 1), mode="replicate")
    dfxdx = F.conv2d(fxp, kx) / dx
    dfydy = F.conv2d(fyp, ky) / dy
    return dfxdx + dfydy


def poisson_op_varcoef(p: torch.Tensor, rho: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """op(p) = div( (1/rho) * grad(p) )"""
    inv_rho = 1.0 / (rho + 1e-6)
    gx, gy = _grad_xy_cd(p, dx, dy)
    fx = inv_rho * gx
    fy = inv_rho * gy
    return _div_xy_cd(fx, fy, dx, dy)


def _inner_mask(mask: torch.Tensor, margin: int = 1) -> torch.Tensor:
    """Zero out a border of width=margin to reduce patch-boundary FD artifacts.
    margin=0 returns a clone of mask.
    """
    m = mask.clone()
    margin = int(max(0, margin))
    if margin == 0:
        return m
    m[:, :, :margin, :] = 0
    m[:, :, -margin:, :] = 0
    m[:, :, :, :margin] = 0
    m[:, :, :, -margin:] = 0
    return m


def _masked_mean_std(field: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """Masked mean/std over spatial dims.

    field/mask: (B,1,H,W)
    Returns:
      mean: (B,1,1,1)
      std : (B,1,1,1)

    NOTE: uses float32 math for numerical stability.
    """
    field = field.float()
    mask = mask.float()
    den = mask.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    mean = (field * mask).sum(dim=(2, 3), keepdim=True) / den
    var = ((field - mean) ** 2 * mask).sum(dim=(2, 3), keepdim=True) / den
    std = torch.sqrt(var + eps)
    return mean, std


def _poisson_scale_from_p_and_grid(
    p_ref: torch.Tensor,
    grid: torch.Tensor,
    *,
    interior_margin: int = 1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Build a dimensionless scaling for Poisson operator residuals.

    Poisson-like operator has typical magnitude ~ p_scale / L^2.
    We estimate:
      p_scale ~ std(p_ref) over masked region
      L      ~ characteristic length inside the patch

    Returns:
      scale: (B,1,1,1) in same units as the operator.
    """
    x = grid[:, 0:1].float()
    y = grid[:, 1:2].float()
    mask = grid[:, 2:3].float()
    m = _inner_mask(mask, margin=interior_margin)

    # masked min/max (avoid contamination by zeros outside the mask)
    big = 1e9
    x_min = x.masked_fill(m < 0.5, big).amin(dim=(2, 3), keepdim=True)
    x_max = x.masked_fill(m < 0.5, -big).amax(dim=(2, 3), keepdim=True)
    y_min = y.masked_fill(m < 0.5, big).amin(dim=(2, 3), keepdim=True)
    y_max = y.masked_fill(m < 0.5, -big).amax(dim=(2, 3), keepdim=True)
    Lx = (x_max - x_min).clamp_min(eps)
    Ly = (y_max - y_min).clamp_min(eps)
    L = 0.5 * (Lx + Ly)

    _, p_std = _masked_mean_std(p_ref, m, eps=eps)
    # Detach: keep this as a *scale*, not a learnable target.
    scale = (p_std / (L * L)).detach().clamp_min(eps)
    return scale


def poisson_residual_loss(
    p_pred: torch.Tensor,
    rho: torch.Tensor,
    rhs_f: torch.Tensor,
    grid: torch.Tensor,   # (B,3,H,W) = [x,y,mask]
    *,
    interior_margin: int = 1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    L_res = || div( (1/rho) grad(p_pred) ) - f ||^2 on the interior masked region.
    """
    x = grid[:, 0:1]
    y = grid[:, 1:2]
    mask = grid[:, 2:3]
    dx, dy = _estimate_dxdy_from_xy(x, y, mask, eps=eps)
    op_pred = poisson_op_varcoef(p_pred, rho, dx, dy)
    m = _inner_mask(mask, margin=interior_margin)
    # IMPORTANT: op(...) has units of Pa/m^2 and can be extremely large when dx,dy are small.
    # We therefore use a *dimensionless* formulation by scaling with a data-driven operator
    # scale ~ std(p)/L^2, where L is the patch physical size inferred from x,y.
    scale = _poisson_scale_from_p_and_grid(p_pred, grid, eps=eps)
    return masked_mse(op_pred / scale, rhs_f / scale, m, eps=eps)


def poisson_consistency_loss(
    p_pred: torch.Tensor,
    p_true: torch.Tensor,
    rho: torch.Tensor,
    grid: torch.Tensor,
    *,
    interior_margin: int = 1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    If you don't have an explicit RHS f, enforce operator consistency:
      || op(p_pred) - op(p_true) ||^2
    """
    x = grid[:, 0:1]
    y = grid[:, 1:2]
    mask = grid[:, 2:3]
    dx, dy = _estimate_dxdy_from_xy(x, y, mask, eps=eps)
    op_pred = poisson_op_varcoef(p_pred, rho, dx, dy)
    op_true = poisson_op_varcoef(p_true, rho, dx, dy)
    m = _inner_mask(mask, margin=interior_margin)
    scale = _poisson_scale_from_p_and_grid(p_true, grid, interior_margin=interior_margin, eps=eps)
    return masked_mse(op_pred / scale, op_true / scale, m, eps=eps)


# =========================
# Optional helper: alpha -> rho (fallback)
# =========================

def get_rho_from_alpha(alpha: torch.Tensor, rho1: float = 1000.0, rho2: float = 1.2) -> torch.Tensor:
    """Simple linear mixture rho = alpha*rho1 + (1-alpha)*rho2."""
    return alpha * float(rho1) + (1.0 - alpha) * float(rho2)

# =========================
# NEW: Poisson loss for interFoam p_rgh equation
#   laplacian(rAUf, p_rgh) = rhs
# where rhs = div(phiHbyA) - Sp_rgh  (as you saved)
# =========================

def poisson_op_rAU(
    p: torch.Tensor,
    rAU: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
) -> torch.Tensor:
    """op(p) = div( rAU_f * grad(p) ) in conservative flux form.

    Why this form:
      OpenFOAM uses a finite-volume laplacian with *face* coefficients (rAUf).
      For structured grids, a good proxy is to build face coefficients by averaging
      adjacent cell-centered rAU, then apply divergence of face fluxes.

    Notes:
      - dx/dy can be scalar (B,1,1,1) or per-pixel (B,1,H,W).
      - Boundary values are left as zero; downstream we apply an interior mask.
    """
    p = p.float()
    rAU = rAU.float()
    dx = dx.float()
    dy = dy.float()

    B, C, H, W = p.shape
    out = torch.zeros_like(p)

    # interior slices
    pc = p[:, :, 1:-1, 1:-1]
    pe = p[:, :, 1:-1, 2:]
    pw = p[:, :, 1:-1, :-2]
    pn = p[:, :, 2:, 1:-1]
    ps = p[:, :, :-2, 1:-1]

    rC = rAU[:, :, 1:-1, 1:-1]
    rE = rAU[:, :, 1:-1, 2:]
    rW = rAU[:, :, 1:-1, :-2]
    rN = rAU[:, :, 2:, 1:-1]
    rS = rAU[:, :, :-2, 1:-1]

    # face coefficients (simple arithmetic average)
    r_e = 0.5 * (rC + rE)
    r_w = 0.5 * (rC + rW)
    r_n = 0.5 * (rC + rN)
    r_s = 0.5 * (rC + rS)

    is_dx_field = (dx.dim() == 4 and dx.shape[-2] == H and dx.shape[-1] == W)
    is_dy_field = (dy.dim() == 4 and dy.shape[-2] == H and dy.shape[-1] == W)

    dx_c = dx[:, :, 1:-1, 1:-1] if is_dx_field else dx
    dy_c = dy[:, :, 1:-1, 1:-1] if is_dy_field else dy

    invDx = 1.0 / dx_c.clamp_min(1e-12)
    invDy = 1.0 / dy_c.clamp_min(1e-12)

    # face fluxes
    fx_e = r_e * (pe - pc) * invDx
    fx_w = r_w * (pc - pw) * invDx
    fy_n = r_n * (pn - pc) * invDy
    fy_s = r_s * (pc - ps) * invDy

    # divergence (conservative)
    div = (fx_e - fx_w) * invDx + (fy_n - fy_s) * invDy

    out[:, :, 1:-1, 1:-1] = div
    return out


def _poisson_scale_from_p_grid_and_coef(
    p_ref: torch.Tensor,
    grid: torch.Tensor,
    coef: torch.Tensor,
    *,
    interior_margin: int = 1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Scale ~ mean(|coef|) * std(p) / L^2."""
    base = _poisson_scale_from_p_and_grid(p_ref, grid, interior_margin=interior_margin, eps=eps)  # std(p)/L^2
    mask = grid[:, 2:3].float()
    m = _inner_mask(mask, margin=interior_margin)
    coef_mean, _ = _masked_mean_std(coef.float().abs(), m, eps=eps)
    return (base * coef_mean.detach()).clamp_min(eps)




def poisson_rAU_residual_loss(
    p_rgh_pred: torch.Tensor,
    rAU: torch.Tensor,
    rhs_f: torch.Tensor,
    grid: torch.Tensor,  # (B,3,H,W) = [x_phys,y_phys,mask]
    *,
    invDx2: Optional[torch.Tensor] = None,
    invDy2: Optional[torch.Tensor] = None,
    interior_margin: int = 1,
    normalize_by_rhs: bool = True,
    rhs_eps_abs: float = 1e-6,
    rhs_max_weight: float = 0.85,
    eps: float = 1e-12,
) -> torch.Tensor:
    x = grid[:, 0:1]
    y = grid[:, 1:2]
    mask = grid[:, 2:3]

    if invDx2 is not None and invDy2 is not None:
        dx = 1.0 / torch.sqrt(invDx2.float().clamp_min(eps))
        dy = 1.0 / torch.sqrt(invDy2.float().clamp_min(eps))
    else:
        dx, dy = _estimate_dxdy_from_xy(x, y, mask, eps=eps)

    op_pred = poisson_op_rAU(p_rgh_pred, rAU, dx, dy)
    m = _inner_mask(mask, margin=interior_margin)

    if bool(normalize_by_rhs):
        res = (op_pred - rhs_f).float()
        rhs_abs = rhs_f.float().abs()
        # Equation (21) in the paper:
        #   s_rhs = max(mean(|rhs|) + 0.85 * max(|rhs|), 1e-6)
        den = m.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
        rhs_mean = (rhs_abs * m).sum(dim=(2, 3), keepdim=True) / den
        rhs_max = (rhs_abs * m).amax(dim=(2, 3), keepdim=True)
        rhs_scale = (rhs_mean + rhs_max * float(rhs_max_weight)).clamp_min(rhs_eps_abs)
        rhs_scale = rhs_scale.detach()
        res_norm = res / rhs_scale
        return masked_mse(res_norm, torch.zeros_like(res_norm), m, eps=eps)

    scale = _poisson_scale_from_p_grid_and_coef(
        p_rgh_pred, grid, rAU, interior_margin=interior_margin, eps=eps
    )
    return masked_mse(op_pred / scale, rhs_f / scale, m, eps=eps)



def poisson_rAU_consistency_loss(
    p_rgh_pred: torch.Tensor,
    p_rgh_true: torch.Tensor,
    rAU: torch.Tensor,
    grid: torch.Tensor,
    *,
    invDx2: Optional[torch.Tensor] = None,
    invDy2: Optional[torch.Tensor] = None,
    interior_margin: int = 1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """If rhs is unavailable, enforce operator consistency:
      || op(p_rgh_pred) - op(p_rgh_true) ||^2
    """
    x = grid[:, 0:1]
    y = grid[:, 1:2]
    mask = grid[:, 2:3]

    if invDx2 is not None and invDy2 is not None:
        dx = 1.0 / torch.sqrt(invDx2.float().clamp_min(eps))
        dy = 1.0 / torch.sqrt(invDy2.float().clamp_min(eps))
    else:
        dx, dy = _estimate_dxdy_from_xy(x, y, mask, eps=eps)

    op_pred = poisson_op_rAU(p_rgh_pred, rAU, dx, dy)
    op_true = poisson_op_rAU(p_rgh_true, rAU, dx, dy)
    m = _inner_mask(mask, margin=interior_margin)
    scale = _poisson_scale_from_p_grid_and_coef(p_rgh_true, grid, rAU, interior_margin=interior_margin, eps=eps)
    return masked_mse(op_pred / scale, op_true / scale, m, eps=eps)



def inner_mask(mask: torch.Tensor, margin: int = 1) -> torch.Tensor:
    """Public wrapper: zero out a border of width=margin.

    Keeping this as a thin wrapper helps downstream code (trainer) pass a configurable
    interior margin without calling private helpers.
    """
    return _inner_mask(mask, margin=margin)


def poisson_rAU_residual_map(
    p_rgh: torch.Tensor,
    rAU: torch.Tensor,
    rhs_f: torch.Tensor,
    grid: torch.Tensor,
    *,
    invDx2: Optional[torch.Tensor] = None,
    invDy2: Optional[torch.Tensor] = None,
    interior_margin: int = 1,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return residual map: op(p_rgh) - rhs and the interior mask used."""
    x = grid[:, 0:1]
    y = grid[:, 1:2]
    mask = grid[:, 2:3]

    if invDx2 is not None and invDy2 is not None:
        dx = 1.0 / torch.sqrt(invDx2.float().clamp_min(eps))
        dy = 1.0 / torch.sqrt(invDy2.float().clamp_min(eps))
    else:
        dx, dy = _estimate_dxdy_from_xy(x, y, mask, eps=eps)

    op = poisson_op_rAU(p_rgh.float(), rAU.float(), dx, dy)
    # Use the same "interior" semantics as the loss / alignment check.
    m = _inner_mask(mask.float(), margin=int(interior_margin))
    return op - rhs_f.float(), m

# =========================
# Gauge / reference losses
# =========================

def gauge_mean_anchor_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Mean-anchor (gauge fixing) loss for pressure-like fields.

    pred/target/mask: (B,1,H,W)

    L = || mean(pred) - mean(target) ||^2  over masked region

    NOTE:
      - force float32 math (AMP-safe)
      - this does NOT constrain gradients, only the constant mode
    """
    pred = pred.float()
    target = target.float()
    mask = mask.float()

    denom = mask.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    mean_pred = (pred * mask).sum(dim=(2, 3), keepdim=True) / denom
    mean_true = (target * mask).sum(dim=(2, 3), keepdim=True) / denom
    return torch.mean((mean_pred - mean_true) ** 2)
