"""Personal DTW calibration candidates and descriptive enrollment coverage checks."""
from __future__ import annotations

import numpy as np

import srdp


BLENDS = (0., .25, .5, .75, 1.)


def pairwise_distances(templates, model):
    pairs = np.zeros((len(templates), len(templates)), dtype=np.float64)
    for i, template in enumerate(templates):
        for j in range(i):
            pairs[i, j] = pairs[j, i] = srdp.dtw_distance(template, templates[j],
                window_ratio=model.dtw_window_ratio, warp_penalty=model.dtw_warp_penalty,
                max_run=model.dtw_max_run)
    return pairs


def baseline_threshold(pairs, calibration_mode, aggregate="min"):
    nearest = np.array([srdp.aggregate_distances(np.delete(row, i).tolist(), aggregate)
                        for i, row in enumerate(pairs)])
    if calibration_mode == "leave_one_recording_out":
        threshold = srdp.calibration_threshold(nearest.tolist(), std_factor=1.)
    else:
        threshold = float(max(nearest.mean() + 2 * nearest.std(), nearest.max() + 1e-6))
    return threshold, nearest


def blended_threshold(baseline, pair_q75, blend):
    if not 0. <= blend <= 1.:
        raise ValueError("DTW calibration blend must be between zero and one")
    return float(baseline + blend * max(0., pair_q75 - baseline))


def pairwise_threshold(pairs, calibration_mode, blend):
    baseline, _ = baseline_threshold(pairs, calibration_mode)
    pair_q75 = float(np.quantile(pairs[np.triu_indices(len(pairs), 1)], .75))
    return blended_threshold(baseline, pair_q75, blend)


def coverage_check(pairs, calibration_mode, aggregate="min"):
    threshold, nearest = baseline_threshold(pairs, calibration_mode, aggregate)
    pending = set(range(len(pairs)))
    components = []
    while pending:
        stack = [min(pending)]
        component = []
        while stack:
            index = stack.pop()
            if index not in pending:
                continue
            pending.remove(index)
            component.append(index)
            stack.extend(other for other in sorted(pending) if pairs[index, other] <= threshold)
        components.append(sorted(component))
    values = pairs[np.triu_indices(len(pairs), 1)]
    return dict(baseline_threshold=threshold, calibration_aggregate=aggregate, held_out_distances=nearest.tolist(),
        loo_passed=(nearest <= threshold).tolist(), components=components,
        pair_min=float(values.min()), pair_q75=float(np.quantile(values, .75)),
        pair_max=float(values.max()), warning=len(components) > 1,
        limitation="Connectivity at the existing threshold is descriptive, not validated phrase or quality rejection")


def select_blend(rows):
    if not rows or not any(r["genuine"] for r in rows) or not any(not r["genuine"] for r in rows):
        raise ValueError("Calibration needs labeled genuine and negative development claims")
    candidates = []
    for blend in BLENDS:
        accepted = [r["voice_passed"] and r["dtw_distance"] <= blended_threshold(
            r["baseline_threshold"], r["pair_q75"], blend) for r in rows]
        baseline = [r["voice_passed"] and r["dtw_distance"] <= r["baseline_threshold"] for r in rows]
        candidates.append(dict(blend=blend,
            genuine_accepted=sum(a and r["genuine"] for a, r in zip(accepted, rows)),
            false_accepts=sum(a and not r["genuine"] for a, r in zip(accepted, rows)),
            new_false_accepts=sum(a and not old and not r["genuine"]
                for a, old, r in zip(accepted, baseline, rows))))
    chosen = min((r for r in candidates if r["new_false_accepts"] == 0),
                 key=lambda r: (-r["genuine_accepted"], r["blend"]))
    return dict(blend=chosen["blend"], rule="No new negative accepts before ASR; maximize genuine accepts; smallest blend wins"), candidates
