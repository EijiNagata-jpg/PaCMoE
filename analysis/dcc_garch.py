"""DCC-GARCH(1,1) (Engle 2002): two-step estimation and one-step-ahead forecasts (Section 3.5).

R. F. Engle (2002), "Dynamic Conditional Correlation: A Simple Class of Multivariate GARCH Models",
JBES 20(3), 339-350. Equation numbers refer to that paper.

    r_t | F_{t-1} ~ N(0, D_t R_t D_t)                                              (25)
    h_{i,t} = omega_i + kappa_i r_{i,t-1}^2 + lambda_i h_{i,t-1}                    (25)
    eps_t = D_t^{-1} r_t
    Q_t = S (1 - a - b) + a eps_{t-1} eps_{t-1}' + b Q_{t-1}                         (23)
    R_t = diag{Q_t}^{-1/2} Q_t diag{Q_t}^{-1/2}                                      (21)
    L = L_V(theta) + L_C(theta, phi)                                                 (27)
    step 1 maximizes L_V over the univariate GARCH parameters, step 2 maximizes L_C over (a, b)  (31)(32)

Implementation choices: S is the unconditional correlation of eps (1/T sum eps eps', normalized to a
unit diagonal); h_{i,0} = mean of r_i^2 in the estimation sample, Q_0 = S; for a series of length T,
h and Q/R have length T+1 and row t (< T) is conditional on F_{t-1}, row T is the one-step forecast.
The parameters, S and h_0 depend only on the sample passed to fit; one_step_forecasts does not re-estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

_LOG2PI = float(np.log(2.0 * np.pi))
_STATIONARITY_MARGIN = 1e-6
_DCC_GRID_A = (0.0, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 0.01, 0.03, 0.05, 0.1)
_DCC_GRID_PERSIST = (0.1, 0.3, 0.5, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995)


# ------------------------------------------------------------------ #
# Step 1: univariate GARCH(1,1)
# ------------------------------------------------------------------ #

def garch11_variance(r: np.ndarray, omega: float, alpha: float, beta: float, h0: float) -> np.ndarray:
    """h_t = omega + alpha r_{t-1}^2 + beta h_{t-1} (t = 1..T), h_0 = h0; length T+1."""
    r = np.asarray(r, dtype=np.float64)
    x = omega + alpha * r ** 2  # x[t-1] is the exogenous term of h_t
    tail, _ = lfilter([1.0], [1.0, -beta], x, zi=[beta * h0])
    return np.concatenate([[h0], tail])


def garch11_loglik(r: np.ndarray, omega: float, alpha: float, beta: float, h0: float) -> float:
    """Term of stock i in Eq. (30)."""
    r = np.asarray(r, dtype=np.float64)
    h = garch11_variance(r, omega, alpha, beta, h0)[:-1]
    if np.any(h <= 0) or not np.all(np.isfinite(h)):
        return -np.inf
    return float(-0.5 * np.sum(_LOG2PI + np.log(h) + r ** 2 / h))


def _persistence_split(s: float, u: float) -> tuple[float, float]:
    """(persistence s, share u) -> (ARCH, GARCH) = (s u, s (1-u)): alpha + beta < 1 with box constraints."""
    return s * u, s * (1.0 - u)


def _grid_then_local(objective, grid, to_x, bounds, n_local=3):
    """Grid search, then L-BFGS-B from the n_local best grid points; returns the best point (as in the MFE
    Toolbox dcc.m). `objective` is the negative log-likelihood of the transformed variables z.
    """
    vals = sorted(((objective(np.asarray(z, dtype=np.float64)), tuple(z)) for z in grid), key=lambda t: t[0])
    best_f, best_z, success = vals[0][0], np.asarray(vals[0][1]), True
    for f0, z0 in vals[:n_local]:
        res = minimize(objective, x0=np.asarray(z0), method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-9})
        if res.fun < best_f:
            best_f, best_z, success = float(res.fun), res.x, bool(res.success)
    return to_x(best_z), best_f, success


_GARCH_GRID_ALPHA = (0.0, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2)
_GARCH_GRID_PERSIST = (0.0, 0.3, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995)


def fit_garch11(r: np.ndarray, h0: float | None = None) -> dict:
    """(Quasi) maximum likelihood GARCH(1,1): omega > 0, alpha, beta >= 0, alpha + beta < 1.

    Variables z = (log omega, s, u), alpha = s u, beta = s (1-u); start points from the variance-targeting
    grid omega = E[r^2](1-s).
    """
    r = np.asarray(r, dtype=np.float64)
    if h0 is None:
        h0 = float(np.mean(r ** 2))
    var = max(float(np.mean(r ** 2)), 1e-12)
    smax = 1.0 - _STATIONARITY_MARGIN

    def to_x(z):
        al, be = _persistence_split(z[1], z[2])
        return float(np.exp(z[0])), al, be

    def nll(z):
        om, al, be = to_x(z)
        ll = garch11_loglik(r, om, al, be, h0)
        return -ll if np.isfinite(ll) else 1e300

    grid = []
    for al in _GARCH_GRID_ALPHA:
        for s in _GARCH_GRID_PERSIST:
            if al <= s:
                s_ = min(s, smax)
                grid.append((np.log(var * max(1.0 - s_, 1e-6)), s_, (al / s_) if s_ > 0 else 0.0))
    bounds = [(np.log(var) - 30.0, np.log(var) + 5.0), (0.0, smax), (0.0, 1.0)]
    (omega, alpha, beta), f, ok = _grid_then_local(nll, grid, to_x, bounds)
    return {"omega": omega, "alpha": alpha, "beta": beta, "h0": float(h0), "loglik": -float(f), "converged": ok}


def volatility_loglik(r: np.ndarray, h: np.ndarray) -> float:
    """Eq. (28)(30): L_V; r, h are (T, n)."""
    r = np.asarray(r, dtype=np.float64)
    return float(-0.5 * np.sum(_LOG2PI + np.log(h) + r ** 2 / h))


# ------------------------------------------------------------------ #
# Step 2: dynamic conditional correlation
# ------------------------------------------------------------------ #

def correlation_target(eps: np.ndarray, kind: str = "correlation") -> np.ndarray:
    """S of Eq. (23): kind="correlation" (Engle 2002) or "covariance" (Q-bar of Engle & Sheppard 2001)."""
    eps = np.asarray(eps, dtype=np.float64)
    M = eps.T @ eps / eps.shape[0]
    if kind == "covariance":
        return M
    if kind != "correlation":
        raise ValueError(f"unknown kind: {kind}")
    d = np.sqrt(np.diag(M))
    return M / np.outer(d, d)


def dcc_recursion(eps: np.ndarray, a: float, b: float, S: np.ndarray, Q0: np.ndarray | None = None):
    """Eqs. (23) and (21); Q, R are (T+1, n, n)."""
    eps = np.asarray(eps, dtype=np.float64)
    T, n = eps.shape
    Q = np.empty((T + 1, n, n))
    Q[0] = S if Q0 is None else Q0
    const = (1.0 - a - b) * S
    for t in range(1, T + 1):
        e = eps[t - 1]
        Q[t] = const + a * np.outer(e, e) + b * Q[t - 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.sqrt(np.einsum("tii->ti", Q))
        R = Q / (d[:, :, None] * d[:, None, :])
    return Q, R


def dcc_correlation_loglik(eps: np.ndarray, a: float, b: float, S: np.ndarray, Q0: np.ndarray | None = None) -> float:
    """Eq. (29): L_C = -1/2 sum_t (log|R_t| + eps_t' R_t^{-1} eps_t - eps_t' eps_t)."""
    eps = np.asarray(eps, dtype=np.float64)
    _, R = dcc_recursion(eps, a, b, S, Q0)
    R = R[:-1]
    with np.errstate(invalid="ignore"):  # -inf when the optimizer tries an infeasible point
        sign, logdet = np.linalg.slogdet(R)
    if np.any(sign <= 0) or not np.all(np.isfinite(logdet)):
        return -np.inf
    quad = np.einsum("ti,ti->t", eps, np.linalg.solve(R, eps[:, :, None])[:, :, 0])
    return float(-0.5 * np.sum(logdet + quad - np.einsum("ti,ti->t", eps, eps)))


def contiguous_pairs(n: int) -> list[tuple[int, int]]:
    """Contiguous pairs (i, i+1) (Pakel et al. 2021; 'Diagonal' in the MFE Toolbox)."""
    return [(i, i + 1) for i in range(n - 1)]


def all_pairs(n: int) -> list[tuple[int, int]]:
    """All pairs (i < j) ('Full' in the MFE Toolbox)."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def dcc_composite_loglik(eps: np.ndarray, a: float, b: float, S: np.ndarray, pairs, Q0: np.ndarray | None = None) -> float:
    """Composite likelihood: sum of bivariate L_C (Eq. 29) over pairs (Pakel, Shephard, Sheppard and Engle 2021).

    The scalar DCC recursion is element-wise, so a pair's Q_t is a sub-matrix of the full Q_t."""
    eps = np.asarray(eps, dtype=np.float64)
    Q0 = S if Q0 is None else Q0
    I = np.array([p[0] for p in pairs])
    J = np.array([p[1] for p in pairs])
    s11, s22, s12 = S[I, I], S[J, J], S[I, J]
    q11, q22, q12 = Q0[I, I].copy(), Q0[J, J].copy(), Q0[I, J].copy()
    c = 1.0 - a - b
    total = 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        for t in range(eps.shape[0]):
            e1, e2 = eps[t, I], eps[t, J]
            rho = q12 / np.sqrt(q11 * q22)
            det = 1.0 - rho ** 2
            if not np.all(det > 0):
                return -np.inf
            total += -0.5 * float(np.sum(np.log(det) + (e1 ** 2 + e2 ** 2 - 2.0 * rho * e1 * e2) / det - (e1 ** 2 + e2 ** 2)))
            q11 = c * s11 + a * e1 * e1 + b * q11
            q22 = c * s22 + a * e2 * e2 + b * q22
            q12 = c * s12 + a * e1 * e2 + b * q12
    return total if np.isfinite(total) else -np.inf


@dataclass
class DCCModel:
    mu: np.ndarray        # mean (0 if mean="zero")
    omega: np.ndarray     # GARCH omega_i
    kappa: np.ndarray     # GARCH ARCH coefficient (kappa_i in Engle 2002)
    lam: np.ndarray       # GARCH GARCH coefficient (lambda_i in Engle 2002)
    h0: np.ndarray
    S: np.ndarray
    Q0: np.ndarray
    a: float
    b: float
    loglik_volatility: float
    loglik_correlation: float   # objective maximized in step 2 (composite likelihood if composite)
    target: str = "correlation"
    method: str = "full"        # "full", "composite-contiguous" or "composite-all"

    def conditional_variances(self, r: np.ndarray) -> np.ndarray:
        """(T+1, n); row t is the variance conditional on F_{t-1}."""
        x = np.asarray(r, dtype=np.float64) - self.mu
        return np.stack([garch11_variance(x[:, i], self.omega[i], self.kappa[i], self.lam[i], self.h0[i])
                         for i in range(x.shape[1])], axis=1)

    def standardized_residuals(self, r: np.ndarray) -> np.ndarray:
        x = np.asarray(r, dtype=np.float64) - self.mu
        return x / np.sqrt(self.conditional_variances(r)[:-1])

    def one_step_forecasts(self, r: np.ndarray):
        """Row t = forecast (h_{t+1}, R_{t+1}) made from r_0..r_t only; no re-estimation."""
        h = self.conditional_variances(r)
        eps = (np.asarray(r, dtype=np.float64) - self.mu) / np.sqrt(h[:-1])
        _, R = dcc_recursion(eps, self.a, self.b, self.S, Q0=self.Q0)
        return h[1:], R[1:]


def fit_dcc(r: np.ndarray, mean: str = "zero", target: str = "correlation",
            method: str = "full", pairs: str = "contiguous") -> DCCModel:
    """Two-step estimation of Eqs. (31)(32); r is (T, n) without missing values. method="composite"
    estimates step 2 by composite likelihood (pairs="contiguous" or "all").
    """
    r = np.asarray(r, dtype=np.float64)
    if np.isnan(r).any():
        raise ValueError("r contains NaN")
    T, n = r.shape
    if mean == "zero":
        mu = np.zeros(n)
    elif mean == "constant":
        mu = r.mean(axis=0)
    else:
        raise ValueError(f"unknown mean: {mean}")
    x = r - mu

    # Step 1 (31): GARCH(1,1) per stock
    fits = [fit_garch11(x[:, i]) for i in range(n)]
    omega = np.array([f["omega"] for f in fits])
    kappa = np.array([f["alpha"] for f in fits])
    lam = np.array([f["beta"] for f in fits])
    h0 = np.array([f["h0"] for f in fits])
    h = np.stack([garch11_variance(x[:, i], omega[i], kappa[i], lam[i], h0[i]) for i in range(n)], axis=1)[:-1]
    eps = x / np.sqrt(h)

    # Step 2 (32): maximize over (a, b) with theta fixed
    S = correlation_target(eps, kind=target)
    Q0 = S.copy()

    smax = 1.0 - _STATIONARITY_MARGIN

    def to_x(z):
        return _persistence_split(z[0], z[1])

    if method == "full":
        def objective(a_, b_):
            return dcc_correlation_loglik(eps, a_, b_, S, Q0)
        method_name = "full"
    elif method == "composite":
        pair_list = {"contiguous": contiguous_pairs, "all": all_pairs}[pairs](n)

        def objective(a_, b_):
            return dcc_composite_loglik(eps, a_, b_, S, pair_list, Q0)
        method_name = f"composite-{pairs}"
    else:
        raise ValueError(f"unknown method: {method}")

    def nll(z):
        ll = objective(*to_x(z))
        return -ll if np.isfinite(ll) else 1e300

    grid = [(min(s_, smax), (a_ / s_) if s_ > 0 else 0.0)
            for a_ in _DCC_GRID_A for s_ in _DCC_GRID_PERSIST if a_ <= s_]
    (a, b), f, _ = _grid_then_local(nll, grid, to_x, [(0.0, smax), (0.0, 1.0)])
    return DCCModel(mu=mu, omega=omega, kappa=kappa, lam=lam, h0=h0, S=S, Q0=Q0, a=a, b=b,
                    loglik_volatility=volatility_loglik(x, h), loglik_correlation=-float(f), target=target, method=method_name)
