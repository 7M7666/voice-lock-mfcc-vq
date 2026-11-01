"""DTW alignment with the project's sum-cost / path-length distance convention."""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _align(local, window_ratio, penalty, max_run):
    n, m = local.shape
    states = 1 if max_run == 0 else 1 + 2 * max_run
    costs = np.full((n + 1, m + 1, states), np.inf)
    parents = np.full((n + 1, m + 1, states), -1, dtype=np.int16)
    costs[0, 0, 0] = 0.0
    constrained = 0 < window_ratio < 1.0
    band = max(abs(n - m), int(round(max(n, m) * window_ratio)))
    for i in range(1, n + 1):
        center = int(round(i * m / n))
        lo = max(1, center - band) if constrained else 1
        hi = min(m, center + band) if constrained else m
        for j in range(lo, hi + 1):
            if max_run == 0:
                options = (costs[i - 1, j, 0] + penalty, costs[i, j - 1, 0] + penalty, costs[i - 1, j - 1, 0])
                choice = np.argmin(np.array(options))
                costs[i, j, 0] = options[choice] + local[i - 1, j - 1]
                parents[i, j, 0] = choice
            else:
                parent = np.argmin(costs[i - 1, j - 1])
                costs[i, j, 0] = costs[i - 1, j - 1, parent] + local[i - 1, j - 1]
                parents[i, j, 0] = parent
                # First vertical/horizontal step may follow a diagonal or the other direction.
                for direction in range(2):
                    first = 1 if direction == 0 else max_run + 1
                    pi, pj = (i - 1, j) if direction == 0 else (i, j - 1)
                    best, parent = costs[pi, pj, 0], 0
                    other = max_run + 1 if direction == 0 else 1
                    for state in range(other, other + max_run):
                        if costs[pi, pj, state] < best:
                            best, parent = costs[pi, pj, state], state
                    costs[i, j, first] = best + penalty + local[i - 1, j - 1]
                    parents[i, j, first] = parent
                    for run in range(1, max_run):
                        state = first + run
                        costs[i, j, state] = costs[pi, pj, state - 1] + penalty + local[i - 1, j - 1]
                        parents[i, j, state] = state - 1
    state = int(np.argmin(costs[n, m]))
    total = costs[n, m, state]
    path = np.empty((n + m, 2), dtype=np.int32)
    length = 0
    i, j = n, m
    if not np.isfinite(total):
        return np.inf, path[:0]
    while i > 0 and j > 0:
        path[length, 0], path[length, 1] = i - 1, j - 1
        length += 1
        parent = int(parents[i, j, state])
        if max_run == 0:
            if parent == 0:
                i -= 1
            elif parent == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        else:
            if state == 0:
                i -= 1
                j -= 1
            elif state <= max_run:
                i -= 1
            else:
                j -= 1
            state = parent
    return total / length, path[:length][::-1]


def alignment(x, y, *, window_ratio=1.0, warp_penalty=0.0, max_run=0):
    if max_run < 0:
        raise ValueError("max_run must be nonnegative")
    if not x.size or not y.size:
        return float("inf"), np.empty((0, 2), dtype=np.int32)
    local = np.stack([np.linalg.norm(y - frame, axis=1) for frame in x])
    return _align(local, window_ratio, warp_penalty, max_run)


def path_statistics(path, n, m):
    steps = np.diff(path, axis=0)
    longest = run = 0
    previous = None
    for step in steps:
        direction = tuple(step)
        run = run + 1 if direction == previous and direction != (1, 1) else int(direction != (1, 1))
        longest = max(longest, run)
        previous = direction
    return {
        "path_length": len(path),
        "length_ratio": len(path) / max(n, m),
        "max_axis_run": longest,
        "diagonal_fraction": float(np.mean(np.all(steps == (1, 1), axis=1))) if len(steps) else 1.0,
        "duration_ratio": n / m,
    }
