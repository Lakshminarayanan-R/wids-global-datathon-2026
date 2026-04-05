'''
This notebook contains the metrics specified in the WiDS competition.
It is a weighted combination of a rank priority metrics (30%) and brier metrics (70%)
'''

import numpy as np
from loguru import logger

def c_index(time, event, risk):
    """
    Calculate the concordance index (C-Index).

    Measures how well the risk scores rank fires by time-to-event.
    For every pair of fires where one hit before the other,
    check if the earlier fire has higher risk.
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    r = np.asarray(risk, dtype=float)
    logger.info(f"{len(t), len(e), len(r)}")

    n = len(t)
    concordant = 0.0
    tied = 0.0
    comparable = 0.0

    for i in range(n):
        if e[i] != 1:  # Only compare when fire i actually hit
            continue
        for j in range(n):
            if i == j or t[i] >= t[j]:  # Fire i must hit before j
                continue
            comparable += 1.0
            if r[i] > r[j]:
                concordant += 1.0
            elif r[i] == r[j]:
                tied += 1.0

    if comparable == 0:
        return 0.5
    return (concordant + 0.5 * tied) / comparable


def brier_at(time, event, prob, H):
    """
    Calculate censoring-aware Brier score at horizon H.

    Excludes samples that were censored before time H
    (because their true outcome is unknown).
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    p = np.clip(np.asarray(prob, dtype=float), 0, 1)

    # Valid samples: not censored before H
    valid = ~((e == 0) & (t < H))

    if valid.sum() == 0:
        return 0.25  # Return baseline if no valid samples

    # True labels: did fire hit by time H?
    y_true = ((e == 1) & (t <= H)).astype(float)[valid]

    # Brier score = mean squared error
    return float(np.mean((p[valid] - y_true) ** 2))


def hybrid_score(time, event, p24, p48, p72, risk=None):
    """
    Calculate the competition hybrid metric.

    Hybrid = 0.3 * C-Index + 0.7 * (1 - Weighted Brier)
    """
    # Use average probability as risk score if not provided
    if risk is None:
        risk = 0.3 * p24 + 0.4 * p48 + 0.3 * p72

    # Calculate C-Index
    ci = c_index(time, event, risk)

    # Calculate weighted Brier
    b24 = brier_at(time, event, p24, 24)
    b48 = brier_at(time, event, p48, 48)
    b72 = brier_at(time, event, p72, 72)
    weighted_brier = 0.3 * b24 + 0.4 * b48 + 0.3 * b72

    # Combine into hybrid score
    hybrid = 0.3 * ci + 0.7 * (1 - weighted_brier)

    return hybrid, ci, weighted_brier