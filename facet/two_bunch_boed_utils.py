"""Shared utilities for two-bunch amortized BOED notebooks."""

import re
from pathlib import Path

import numpy as np
import torch
from xopt.generator import Generator
from xopt.errors import VOCSError
from pydantic import (
    ValidationInfo,
    field_validator,
)

# ── Normalization constants ────────────────────────────────────────────────────
# [A, f1, tau1, tau2, tau_kick, sigma, y_shift, irf_sigma, alpha_kick, delta_T]

_NORM_EPS = 1e-6

_NUISANCE_LOW  = [1.5, 0.1, 1.0,  15.0, 0.1,    0.0001, -0.05, 0.01, 0.0, 0.5]
_NUISANCE_HIGH = [10., 0.9, 15.0, 100., 10.0,   0.01,    0.05,  3.0, 0.4, 45.]
_LOG_UNIFORM_IDX = [2, 3, 4, 9]   # tau1, tau2, tau_kick, delta_T


# ── Simulator ──────────────────────────────────────────────────────────────────

class TwoBunchDoubleExp:
    """Double-exponential two-bunch simulator with Gaussian IRF convolution.

    nuisance: [A, f1, tau1, tau2, tau_kick, sigma, y_shift, irf_sigma, alpha_kick, delta_T]
    """

    def __init__(self, noise_type='gaussian'):
        self.noise_type = noise_type

    def __call__(self, xi, thetas, nuisance, noiseless=False):
        """xi: (N,1), thetas: (B,1), nuisance: (B,10) → (N,1)"""
        if nuisance.dim() == 1: nuisance = nuisance.unsqueeze(0)
        if thetas.dim()  == 1: thetas   = thetas.unsqueeze(0)
        if xi.dim()      == 1: xi       = xi.unsqueeze(1)

        A          = nuisance[:, :1]
        f1         = nuisance[:, 1:2]
        tau1       = nuisance[:, 2:3]
        tau2       = nuisance[:, 3:4]
        tau_kick   = nuisance[:, 4:5]
        sigma      = nuisance[:, 5:6]
        y_shift    = nuisance[:, 6:7]
        irf_sigma  = nuisance[:, 7:8]
        alpha_kick = nuisance[:, 8:9]
        delta_T    = nuisance[:, 9:10]

        n_xi = xi.shape[0]
        if A.shape[0] != n_xi:
            A, f1, tau1, tau2, tau_kick = [v.repeat(n_xi, 1) for v in (A, f1, tau1, tau2, tau_kick)]
            sigma, y_shift, irf_sigma   = [v.repeat(n_xi, 1) for v in (sigma, y_shift, irf_sigma)]
            alpha_kick, delta_T, thetas = [v.repeat(n_xi, 1) for v in (alpha_kick, delta_T, thetas)]

        xi = xi.view(-1, 1)
        s = irf_sigma
        sqrt2 = xi.new_tensor(2.0).sqrt()

        def exp_component(amp, tau_i, dt):
            q  = (s**2 / tau_i - dt) / (s * sqrt2)
            b1 = (amp / 2) * torch.exp(-(dt**2) / (2 * s**2)) * torch.special.erfcx(q.clamp(min=0))
            b2 = (amp / 2) * torch.exp((s**2 / (2 * tau_i**2) - dt / tau_i).clamp(max=0)) * torch.special.erfc(q.clamp(max=0))
            return torch.where(q >= 0, b1, b2)

        dt1 = xi - thetas
        dt2 = xi - (thetas + delta_T)

        step_fast = -(f1 * A / 2)          * (1 + torch.special.erf(dt1 / (s * sqrt2)))
        step_slow = -((1 - f1) * A / 2)    * (1 + torch.special.erf(dt1 / (s * sqrt2)))
        step_kick = -(alpha_kick * A / 2)   * (1 + torch.special.erf(dt2 / (s * sqrt2)))

        y_out = (
            -(exp_component(f1 * A,          tau1,     dt1) + step_fast)
            -(exp_component((1 - f1) * A,    tau2,     dt1) + step_slow)
            -(exp_component(alpha_kick * A,  tau_kick, dt2) + step_kick)
            + y_shift
        )

        if not noiseless:
            if self.noise_type == 'gaussian':
                y_out = y_out + sigma * torch.randn_like(y_out)
            else:
                y_out = y_out + (2 * sigma * torch.rand_like(y_out) - sigma)
        return y_out


# ── Normalizer ─────────────────────────────────────────────────────────────────

class GridScanNormalizer:
    def __init__(self, t_min, t_max, y_min, y_max, eps=_NORM_EPS):
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.eps   = eps

    def normalize_x(self, t):
        return (t - self.t_min) / (self.t_max - self.t_min)

    def normalize_y(self, y):
        return (y - self.y_min) / (self.y_max - self.y_min + self.eps)

    def denormalize_t0(self, t0_norm):
        return t0_norm * (self.t_max - self.t_min) + self.t_min


# ── Nuisance helpers ───────────────────────────────────────────────────────────

def denormalize_nuisance(nu_norm):
    """nu_norm: (..., 10) in [0,1] → physical units."""
    low  = torch.tensor(_NUISANCE_LOW,  dtype=torch.float32)
    high = torch.tensor(_NUISANCE_HIGH, dtype=torch.float32)
    nu   = nu_norm.float()
    out  = nu * (high - low) + low
    for i in _LOG_UNIFORM_IDX:
        out[..., i] = torch.exp(
            nu[..., i] * (torch.log(high[i]) - torch.log(low[i])) + torch.log(low[i])
        )
    return out


def find_full_param_model(model_dir):
    matches = sorted(Path(model_dir).glob('full_param_posterior_traced_horizon_*.pt'))
    return matches[-1] if matches else None


@torch.no_grad()
def compute_predictive_curves(t0_phys, nu_phys, t_fine, t_grid_phys, sim, n_plot):
    """Simulate noiseless predictive curves, each normalized by its own grid-scan y_min/y_max."""
    K      = min(n_plot, len(t0_phys))
    n_fine = t_fine.shape[0]
    n_grid = t_grid_phys.shape[0]
    t0_t   = torch.tensor(t0_phys[:K], dtype=torch.float32).unsqueeze(1)
    nu_t   = torch.tensor(nu_phys[:K], dtype=torch.float32)

    y_fine = sim(
        t_fine.repeat(K, 1),
        t0_t.repeat_interleave(n_fine, 0),
        nu_t.repeat_interleave(n_fine, 0),
        noiseless=True,
    ).cpu().view(K, n_fine).numpy()

    y_grid = sim(
        t_grid_phys.repeat(K, 1),
        t0_t.repeat_interleave(n_grid, 0),
        nu_t.repeat_interleave(n_grid, 0),
        noiseless=True,
    ).cpu().view(K, n_grid).numpy()

    y_min = y_grid.min(axis=1, keepdims=True)
    y_max = y_grid.max(axis=1, keepdims=True)
    return (y_fine - y_min) / (y_max - y_min + _NORM_EPS)


# ── Generator ──────────────────────────────────────────────────────────────────

class AmortizedBOEDBunchGenerator(Generator):
    """Amortized BOED generator for two-bunch T0 search.

    Phase 1 (n_obs < grid_steps): returns evenly-spaced grid points.
    Phase 2 (n_obs >= grid_steps): samples from GMM posterior models.
      Normalization is fitted automatically from the grid-scan data on the
      first BOED call; no manual fit_normalization() step is required.

    Normalization:
        x_norm = (t_phys - t_min) / (t_max - t_min)
        y_norm = (y - y_min) / (y_max - y_min + eps)   [y_min/y_max from grid scan]
    """

    device:     str = 'cpu'
    grid_steps: int = 10
    block_size: int = 10

    @field_validator("vocs", mode="after")
    def validate_vocs(cls, v, info: ValidationInfo):
        if v.n_constraints > 0 and not info.data["supports_constraints"]:
            raise VOCSError("this generator does not support constraints")

        # assert that the generator had no objectives
        if not v.n_objectives == 0:
            raise VOCSError("AmortizedBOEDBunchGenerator generator only supports problems with no objectives")

        return v

    def __init__(self, model_dir, design_range, observable_name, vocs=None, **kwargs):
        super().__init__(vocs=vocs, **kwargs)
        model_dir = Path(model_dir)
        t_min, t_max = float(design_range[0]), float(design_range[1])

        pattern = re.compile(r'posterior_sampling_traced_round_(\d+)_horizon_\d+\.pt$')
        round_models = {}
        for p in sorted(model_dir.glob('posterior_sampling_traced_round_*_horizon_*.pt')):
            m = pattern.match(p.name)
            if m:
                round_models[int(m.group(1))] = p

        if not round_models:
            raise FileNotFoundError(f'No round models found in {model_dir}')

        self.__dict__.update({
            '_round_to_path':     round_models,
            '_available_rounds':  sorted(round_models.keys()),
            '_model_cache':       {},
            '_normalizer':        None,
            '_t_min':             t_min,
            '_t_max':             t_max,
            '_grid_pts':          list(np.linspace(t_min, t_max, self.grid_steps)),
            '_observable':        observable_name,
            '_last_round':        None,
        })
        print(f'Available rounds: {self._available_rounds}')
        print(f'Design range: [{t_min}, {t_max}]°')

    def _load_model(self, r):
        if r not in self._model_cache:
            model = torch.jit.load(str(self._round_to_path[r]), map_location=self.device)
            model.eval()
            self._model_cache[r] = model
        return self._model_cache[r]

    def _select_round(self, n_obs):
        target   = (n_obs - self.grid_steps) // self.block_size
        eligible = [r for r in self._available_rounds if r <= target]
        return max(eligible) if eligible else self._available_rounds[0]

    def generate(self, n_candidates=1):
        var   = self.vocs.variable_names[0]
        n_obs = len(self.data) if (self.data is not None and len(self.data) > 0) else 0

        # Grid phase
        if n_obs < self.grid_steps:
            return [{var: self._grid_pts[n_obs]} for _ in range(n_candidates)]

        # Auto-fit normalization from grid-scan data on first BOED call
        if self._normalizer is None:
            y = self.data[self._observable].values
            self.__dict__['_normalizer'] = GridScanNormalizer(
                t_min=self._t_min, t_max=self._t_max,
                y_min=float(y.min()), y_max=float(y.max()),
            )
            print(f'Normalization auto-fitted: y_min={self._normalizer.y_min:.4f}  y_max={self._normalizer.y_max:.4f}')

        norm   = self._normalizer
        t_phys = torch.tensor(self.data[var].values,              dtype=torch.float32).unsqueeze(1)
        y_phys = torch.tensor(self.data[self._observable].values, dtype=torch.float32).unsqueeze(1)
        xi_norm = norm.normalize_x(t_phys).unsqueeze(0)  # (1, T, 1)
        y_norm  = norm.normalize_y(y_phys).unsqueeze(0)  # (1, T, 1)

        r = self._select_round(n_obs)
        if r != self._last_round:
            print(f'  → round {r:02d} model active at n_obs={n_obs}')
            self.__dict__['_last_round'] = r

        with torch.no_grad():
            means, stds, log_weights = self._load_model(r)(xi_norm, y_norm)

        weights      = torch.softmax(log_weights.squeeze(0), dim=-1)
        idx          = torch.multinomial(weights, n_candidates, replacement=True)
        samples_norm = (
            means[0, idx, :] + stds[0, idx, :] * torch.randn(n_candidates, means.shape[-1])
        )[:, 0].clamp(0., 1.)
        t_proposed = norm.denormalize_t0(samples_norm)
        return [{var: float(t)} for t in t_proposed]
