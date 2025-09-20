"""Minimum-norm Pareto weighting utilities.

This module offers a self-contained implementation of the Pareto-constrained
quadratic programme described in the accompanying manuscript excerpt.  The
design goals were:

* handle gradient inputs from Python lists or (optionally) torch tensors;
* compute minimum-norm convex combinations on the simplex without relying on
  heavy dependencies such as NumPy (which is not available in the execution
  environment);
* provide the closed-form active-set routine for up to three tasks and a
  projected-gradient / OSQP hybrid for larger task counts;
* expose simple diagnostics so that downstream scripts can monitor degeneracy
  handling and fallback usage.

The implementation deliberately sticks to pure Python linear algebra so that it
can execute in minimal environments while still mirroring the logic discussed
in the paper.  When OSQP is available it will be used, otherwise the solver
falls back to a numerically stable projected gradient descent on the simplex.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Sequence, Tuple, Union

try:  # Optional dependency for accelerated path when present.
    import torch
except Exception:  # pragma: no cover - torch may not be installed.
    torch = None  # type: ignore

try:
    import osqp  # type: ignore
    from scipy import sparse  # type: ignore
    _OSQP_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency.
    osqp = None  # type: ignore
    sparse = None  # type: ignore
    _OSQP_AVAILABLE = False


Vector = List[float]
Matrix = List[List[float]]
ArrayLike = Union[Sequence[float], "torch.Tensor"]


def _to_vector(array: ArrayLike) -> Vector:
    """Flatten gradients into plain Python lists of floats."""

    if torch is not None and isinstance(array, torch.Tensor):
        flat = array.detach().cpu().reshape(-1).tolist()
        return [float(x) for x in flat]
    return [float(x) for x in array]  # type: ignore[arg-type]


def _stack_gradients(gradients: Sequence[ArrayLike]) -> List[Vector]:
    if len(gradients) == 0:
        raise ValueError("Expected at least one gradient vector.")
    return [_to_vector(g) for g in gradients]


def _dot(u: Vector, v: Vector) -> float:
    return sum(x * y for x, y in zip(u, v))


def _norm(u: Vector) -> float:
    return _dot(u, u) ** 0.5


def _scale(u: Vector, scalar: float) -> Vector:
    return [scalar * x for x in u]


def _add(u: Vector, v: Vector) -> Vector:
    return [x + y for x, y in zip(u, v)]


def _normalise_gradients(
    gradients: List[Vector], epsilon: float
) -> Tuple[List[Vector], List[float]]:
    normalised: List[Vector] = []
    norms: List[float] = []
    for g in gradients:
        norm = _norm(g)
        norms.append(norm)
        if norm <= epsilon:
            normalised.append([0.0 for _ in g])
        else:
            inv = 1.0 / norm
            normalised.append([x * inv for x in g])
    return normalised, norms


def _gram_matrix(gradients: List[Vector]) -> Matrix:
    k = len(gradients)
    gram: Matrix = [[0.0 for _ in range(k)] for _ in range(k)]
    for i in range(k):
        for j in range(i, k):
            value = _dot(gradients[i], gradients[j])
            gram[i][j] = value
            gram[j][i] = value
    return gram


def _matrix_copy(matrix: Matrix) -> Matrix:
    return [row[:] for row in matrix]


def _matrix_identity(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _matrix_infinity_norm(matrix: Matrix) -> float:
    return max(sum(abs(x) for x in row) for row in matrix)


def _gaussian_elimination(matrix: Matrix, rhs: Vector, epsilon: float) -> Vector:
    n = len(matrix)
    aug = [matrix[i][:] + [rhs[i]] for i in range(n)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) <= epsilon:
            raise ValueError("Matrix is singular.")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        pivot_val = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot_val

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    return [aug[i][-1] for i in range(n)]


def _invert_matrix(matrix: Matrix, epsilon: float) -> Matrix:
    n = len(matrix)
    aug = [matrix[i][:] + identity_row[:] for i, identity_row in enumerate(_matrix_identity(n))]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) <= epsilon:
            raise ValueError("Matrix is singular.")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        pivot_val = aug[col][col]
        for j in range(col, 2 * n):
            aug[col][j] /= pivot_val

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            for j in range(col, 2 * n):
                aug[row][j] -= factor * aug[col][j]

    return [[aug[i][n + j] for j in range(n)] for i in range(n)]


def _condition_number(matrix: Matrix, epsilon: float) -> float:
    try:
        inverse = _invert_matrix(matrix, epsilon)
    except ValueError:
        return float("inf")
    return _matrix_infinity_norm(matrix) * _matrix_infinity_norm(inverse)


def _project_to_simplex(vector: Vector, z: float = 1.0) -> Vector:
    if z <= 0:
        raise ValueError("Simplex radius must be positive.")
    if not vector:
        return []

    u = sorted(vector, reverse=True)
    cssv = []
    total = 0.0
    for val in u:
        total += val
        cssv.append(total)
    rho = -1
    for i, (val, csum) in enumerate(zip(u, cssv)):
        if val * (i + 1) > csum - z:
            rho = i
    if rho == -1:
        theta = 0.0
    else:
        theta = (cssv[rho] - z) / (rho + 1)
    return [max(x - theta, 0.0) for x in vector]


def _expand_solution(lam: Vector, active: Sequence[int], total: int) -> Vector:
    full = [0.0] * total
    for idx, weight in zip(active, lam):
        full[idx] = weight
    return full


def _objective(gram: Matrix, lam: Vector) -> float:
    k = len(lam)
    total = 0.0
    for i in range(k):
        row_sum = sum(gram[i][j] * lam[j] for j in range(k))
        total += lam[i] * row_sum
    return total


def _solve_subset(
    gram: Matrix,
    subset: Tuple[int, ...],
    epsilon: float,
    cond_threshold: float,
) -> Optional[Vector]:
    n = len(subset)
    reduced = [[gram[i][j] for j in subset] for i in subset]
    if _condition_number(reduced, epsilon) >= cond_threshold:
        return None
    ones = [1.0] * n
    try:
        solution = _gaussian_elimination(
            [[reduced[i][j] + (epsilon if i == j else 0.0) for j in range(n)] for i in range(n)],
            ones,
            epsilon,
        )
    except ValueError:
        return None

    denom = sum(solution)
    if abs(denom) <= epsilon:
        return None
    lam_subset = [max(x / denom, 0.0) for x in solution]
    if any(x < -1e-7 for x in lam_subset):
        return None

    k = len(gram)
    lam_full = [0.0] * k
    for idx, value in zip(subset, lam_subset):
        lam_full[idx] = value
    total = sum(lam_full)
    if total <= epsilon:
        return None
    return [x / total for x in lam_full]


def _closed_form_min_norm(gram: Matrix, epsilon: float, cond_threshold: float) -> Optional[Vector]:
    k = len(gram)
    best: Optional[Vector] = None
    best_obj = float("inf")

    for r in range(1, k + 1):
        for subset in combinations(range(k), r):
            lam = _solve_subset(gram, subset, epsilon, cond_threshold)
            if lam is None:
                continue
            obj = _objective(gram, lam)
            if obj < best_obj:
                best_obj = obj
                best = lam
    return best


def _attempt_osqp(
    gram: Matrix,
    epsilon: float,
    residual_tol: float,
    max_iter: int,
    alpha: float,
    rho: float,
    warm_start: Optional[Vector],
) -> Optional[Vector]:
    if not _OSQP_AVAILABLE:
        return None

    import numpy as np  # Local import to avoid mandatory dependency.

    k = len(gram)
    gram_np = np.array(gram, dtype=float) + epsilon * np.eye(k)
    q = np.zeros(k)
    A = sparse.vstack(
        [sparse.csc_matrix(np.ones((1, k))), sparse.identity(k, format="csc")]
    )
    l = np.concatenate([np.array([1.0]), np.zeros(k)])
    u = np.concatenate([np.array([1.0]), np.full(k, np.inf)])

    solver = osqp.OSQP()
    solver.setup(
        P=sparse.csc_matrix(gram_np),
        q=q,
        A=A,
        l=l,
        u=u,
        max_iter=max_iter,
        eps_abs=residual_tol,
        eps_rel=residual_tol,
        alpha=alpha,
        rho=rho,
        verbose=False,
    )

    if warm_start is not None:
        solver.warm_start(x=np.array(warm_start, dtype=float))

    result = solver.solve()
    if result.info.status_val not in (1, 2):  # solved or solved inaccurate
        return None
    if max(abs(result.info.pri_res), abs(result.info.dual_res)) > residual_tol:
        return None

    lam = [max(float(x), 0.0) for x in result.x]
    total = sum(lam)
    if total <= epsilon:
        return None
    return [x / total for x in lam]


def _projected_gradient_fallback(
    gram: Matrix,
    epsilon: float,
    initial: Optional[Vector],
    max_iter: int = 250,
    tol: float = 1e-6,
) -> Vector:
    k = len(gram)
    lam = initial[:] if initial is not None else [1.0 / k] * k

    def matrix_vector(mat: Matrix, vec: Vector) -> Vector:
        return [sum(mat[i][j] * vec[j] for j in range(k)) for i in range(k)]

    L = max(sum(abs(x) for x in row) for row in gram)
    step = 1.0 / (L if L > 0 else k)

    for _ in range(max_iter):
        grad = matrix_vector(gram, lam)
        candidate = _project_to_simplex([lam[i] - step * grad[i] for i in range(k)])
        diff = sum(abs(candidate[i] - lam[i]) for i in range(k))
        lam = candidate
        if diff < tol:
            break

    total = sum(lam)
    if total <= epsilon:
        return [1.0 / k] * k
    return [x / total for x in lam]


@dataclass
class ParetoQPStats:
    degeneracy_detected: bool = False
    used_osqp: bool = False
    fallback_used: bool = False
    osqp_available: bool = _OSQP_AVAILABLE


class ParetoQP:
    def __init__(
        self,
        epsilon: float = 1e-8,
        cond_threshold: float = 1e8,
        residual_tol: float = 1e-4,
        max_iter: int = 50,
        alpha: float = 1.6,
        rho: float = 0.1,
    ) -> None:
        self.epsilon = epsilon
        self.cond_threshold = cond_threshold
        self.residual_tol = residual_tol
        self.max_iter = max_iter
        self.alpha = alpha
        self.rho = rho
        self.prev_lambda: Optional[Vector] = None
        self.stats = ParetoQPStats()

    def solve(self, gradients: Sequence[ArrayLike]) -> Vector:
        stacked = _stack_gradients(gradients)
        normalised, norms = _normalise_gradients(stacked, self.epsilon)

        total_tasks = len(gradients)
        active = [idx for idx, norm in enumerate(norms) if norm > self.epsilon]
        stats = ParetoQPStats()

        if not active:
            stats.fallback_used = True
            uniform = [1.0 / total_tasks] * total_tasks
            self.prev_lambda = uniform
            self.stats = stats
            return uniform

        active_gradients = [normalised[i] for i in active]
        gram = _gram_matrix(active_gradients)
        condition = _condition_number(gram, self.epsilon)
        if condition >= self.cond_threshold:
            stats.degeneracy_detected = True
            stats.fallback_used = True
            lam_active = [1.0 / len(active)] * len(active)
            lam_full = _expand_solution(lam_active, active, total_tasks)
            self.prev_lambda = lam_full
            self.stats = stats
            return lam_full

        warm_start: Optional[Vector] = None
        if self.prev_lambda is not None and len(self.prev_lambda) == total_tasks:
            warm_start = [self.prev_lambda[i] for i in active]
            total_prev = sum(warm_start)
            if total_prev > self.epsilon:
                warm_start = [w / total_prev for w in warm_start]
            else:
                warm_start = None

        active_k = len(active)
        if active_k <= 3:
            lam_active = _closed_form_min_norm(gram, self.epsilon, self.cond_threshold)
            if lam_active is None:
                stats.fallback_used = True
                lam_active = [1.0 / active_k] * active_k
        else:
            lam_active = _attempt_osqp(
                gram=gram,
                epsilon=self.epsilon,
                residual_tol=self.residual_tol,
                max_iter=self.max_iter,
                alpha=self.alpha,
                rho=self.rho,
                warm_start=warm_start,
            )
            if lam_active is not None:
                stats.used_osqp = True
            else:
                stats.fallback_used = True
                lam_active = _projected_gradient_fallback(
                    gram=gram,
                    epsilon=self.epsilon,
                    initial=warm_start,
                )

        lam_full = _expand_solution(lam_active, active, total_tasks)
        self.prev_lambda = lam_full
        self.stats = stats
        return lam_full

    def combine(
        self, gradients: Sequence[ArrayLike]
    ) -> Tuple[Union[Vector, "torch.Tensor"], Vector]:
        lam = self.solve(gradients)
        first = gradients[0]
        if torch is not None and isinstance(first, torch.Tensor):
            lam_tensor = torch.tensor(lam, dtype=first.dtype, device=first.device)
            combined = sum(lam_tensor[i] * g for i, g in enumerate(gradients))
            return combined, lam
        stacked = _stack_gradients(gradients)
        combined = [0.0 for _ in stacked[0]]
        for weight, grad in zip(lam, stacked):
            combined = _add(combined, _scale(grad, weight))
        return combined, lam


def minimum_norm_combination(
    gradients: Sequence[ArrayLike], solver: Optional[ParetoQP] = None
) -> Tuple[Union[Vector, "torch.Tensor"], Vector]:
    solver = solver or ParetoQP()
    return solver.combine(gradients)


__all__ = ["ParetoQP", "ParetoQPStats", "minimum_norm_combination"]

