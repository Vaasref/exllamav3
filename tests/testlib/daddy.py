"""Small, independent CPU oracle for DADDy (not a production sampler)."""
import numpy as np
from testlib.sampling import element_noise

BOUNDARIES = np.array([0.03, 0.08, 0.15, 0.23, 0.32, 0.45, 0.65])


def distribution(logits, mode=1, *, inv_temp=1., minp_log=0., filters=0,
                 top_k=0, top_p=1., inv_temp_filter=1.):
    x = np.asarray(logits, dtype=np.float64)
    eligible = np.isfinite(x)
    if not eligible.any():
        return np.zeros_like(x)
    maximum = x[eligible].max()
    if mode == 0:
        probs = np.zeros_like(x)
        probs[np.argmax(x)] = 1.
        return probs
    if mode == 2 or (mode == 3 and filters & 4):
        eligible &= x >= maximum + minp_log
    bias = np.zeros_like(x)
    if mode == 3:
        if filters & 1:
            # Match the existing fused kernel's documented 32-nat histogram range.
            ordered = np.sort(x[eligible])[::-1]
            if top_k < len(ordered):
                threshold = ordered[top_k - 1]
                eligible &= x >= threshold
            eligible &= (maximum - x) * inv_temp_filter < 32.
        if filters & 2:
            idx = np.flatnonzero(eligible)
            idx = idx[np.argsort(-x[idx], kind='stable')]
            weights = np.exp((x[idx] - maximum) * inv_temp_filter)
            crossing = min(np.searchsorted(np.cumsum(weights), top_p * weights.sum(), side='left'), len(idx) - 1)
            threshold = x[idx[crossing]]
            ties = eligible & (x == threshold)
            retained_ties = crossing + 1 - np.count_nonzero(x[idx] > threshold)
            bias[ties] = np.log(retained_ties / np.count_nonzero(ties))
            eligible &= x >= threshold
    log_weights = (x - maximum) * inv_temp + bias
    weights = np.where(eligible, np.exp(log_weights), 0.)
    return weights / weights.sum()


def step(probs, balances, seed, proposal=None, *, noise=None):
    """Mutates balances once; accepts an already prepared eligible distribution."""
    probs = np.asarray(probs, dtype=np.float64)
    eligible = probs > 0
    if not eligible.any():
        return 0, dict(path='empty', bin=None, mass=np.zeros(8))
    probs = probs / probs.sum()
    bins = np.searchsorted(BOUNDARIES, probs / probs.max(), side='right')
    mass = np.bincount(bins, weights=probs, minlength=8)
    mass /= mass.sum()
    before = balances.copy()
    balances += mass
    present = np.bincount(bins[eligible], minlength=8) > 0
    if proposal is not None and 0 <= proposal < len(probs) and eligible[proposal] and balances[bins[proposal]] > 0:
        selected, token, path = bins[proposal], proposal, 'proposal'
    else:
        remainder = seed & 3
        order = list(range(7, -1, -1) if remainder == 0 else range(8))
        open_bins = [b for b in order if present[b] and balances[b] > 0]
        path = 'draw' if open_bins else 'all_full'
        selected = open_bins[0] if open_bins else max((b for b in order if present[b]), key=lambda b: balances[b])
        if noise is None:
            _, noise = element_noise(seed, len(probs))
        with np.errstate(divide='ignore'):
            objective = np.where(eligible & (bins == selected), np.log(probs) + noise, -np.inf)
        token = int(np.argmax(objective))
    if np.count_nonzero(present) == 1:
        balances[:] = before
    else:
        balances[selected] -= 1
    return int(token), dict(path=path, bin=int(selected), mass=mass)
