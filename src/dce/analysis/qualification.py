"""Amendment v1.9 §2.3 (D-036): qualification of a rotation × offset pair on validation.

A logistic regression of f_new's decision (sign of logit_1 − logit_0, i.e. sign of m_n for class 1) on the
five factors gives w and b; then α̂ = w_colour / w_aspect and t̂ is the aspect value on the fitted boundary
with colour and every other factor at 0.5. The fit is unpenalised maximum likelihood (Newton–IRLS with step
halving); `separable` flags data that a linear rule splits perfectly, where the MLE does not exist.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

ASPECT, COLOUR = 0, 2


@dataclass(frozen=True)
class LogitFit:
    w: list[float]
    b: float
    n_iter: int
    converged: bool
    separable: bool
    train_accuracy: float


def _nll(z: np.ndarray, s: np.ndarray, ridge: float, w: np.ndarray) -> float:
    return float(np.sum(np.logaddexp(0.0, z) - s * z) + 0.5 * ridge * w[1:] @ w[1:])


def fit_logistic(x: np.ndarray, s: np.ndarray, ridge: float = 0.0, max_iter: int = 200, tol: float = 1e-10) -> LogitFit:
    """Logistic regression s ~ σ(b + x w); `ridge` (default 0) penalises w only, for sensitivity checks."""
    x = np.asarray(x, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    a = np.c_[np.ones(len(x)), x]
    theta = np.zeros(a.shape[1])
    reg = np.full(a.shape[1], ridge)
    reg[0] = 0.0
    f = _nll(a @ theta, s, ridge, theta)
    converged, it = False, 0
    for it in range(1, max_iter + 1):
        z = a @ theta
        p = 1.0 / (1.0 + np.exp(-z))
        g = a.T @ (p - s) + reg * theta
        h = (a * (p * (1 - p))[:, None]).T @ a + np.diag(reg) + 1e-12 * np.eye(a.shape[1])
        step = np.linalg.solve(h, g)
        lr = 1.0
        while lr > 1e-8:
            cand = theta - lr * step
            fc = _nll(a @ cand, s, ridge, cand)
            if fc <= f:
                break
            lr /= 2
        if fc > f:          # no descent along the Newton step: stalled at the optimum to numerical precision
            converged = True
            break
        theta, f_old, f = cand, f, fc
        if abs(f_old - f) <= tol * max(1.0, abs(f)):
            converged = True
            break
    acc = float(((a @ theta > 0) == (s > 0.5)).mean())
    return LogitFit(w=theta[1:].tolist(), b=float(theta[0]), n_iter=it, converged=converged,
                    separable=acc == 1.0, train_accuracy=acc)


def boundary_params(fit: LogitFit) -> tuple[float, float]:
    """(α̂, t̂) from a fit: α̂ = w_c / w_a; t̂ solves b + w_a t̂ + 0.5 Σ_{k≠aspect} w_k = 0."""
    w = np.asarray(fit.w)
    alpha_hat = float(w[COLOUR] / w[ASPECT])
    t_hat = float(-(fit.b + 0.5 * (w.sum() - w[ASPECT])) / w[ASPECT])
    return alpha_hat, t_hat


def qualify(alpha: float, t: float, alpha_hat: float, t_hat: float, accuracy: float) -> dict:
    """The three conditions of v1.9 §2.3 (non-null cells)."""
    c1 = abs(alpha_hat - alpha) <= 0.05 + 0.3 * alpha
    c2 = abs(t_hat - t) <= 0.02
    c3 = accuracy >= 0.90
    return {"alpha_ok": bool(c1), "t_ok": bool(c2), "accuracy_ok": bool(c3), "qualified": bool(c1 and c2 and c3)}


def qualification_row(phi: np.ndarray, decision_new: np.ndarray, y_rule: np.ndarray, alpha: float, t: float,
                      ridge: float = 0.0) -> dict:
    fit = fit_logistic(phi, decision_new, ridge=ridge)
    a_hat, t_hat = boundary_params(fit)
    acc = float((decision_new == y_rule).mean())
    return {"alpha": alpha, "t": t, "alpha_hat": a_hat, "t_hat": t_hat, "accuracy_new_rule": acc,
            **qualify(alpha, t, a_hat, t_hat, acc), "fit": asdict(fit), "ridge": ridge}
