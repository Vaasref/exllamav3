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


def fallback_weights(balances, present, proposal_bin=None, *, proposal_given=False,
                     overdraft=0., boost=.5, fixed_repulsor=.5, dynamic_repulsor=.75):
    """Policy reference independent of the GPU's fused Gumbel reduction."""
    credit = np.where(present, np.maximum(balances + overdraft, 0.), 0.)
    if not credit.any():
        greatest = np.max(balances[present])
        return (present & (balances == greatest)).astype(float), 'all_full'
    if not proposal_given:
        return credit, 'draw'
    shares = credit / credit.sum()
    weights = shares + boost * np.minimum(shares, .125)
    for center, strength in [(2, fixed_repulsor), (proposal_bin, dynamic_repulsor)]:
        if center is not None:
            distance = np.abs(np.arange(8) - center)
            weights *= 1 - strength * np.where(distance <= 2, 0.5 ** distance, 0.)
    return (weights if weights.any() else credit), 'draw'


def step(probs, balances, seed, proposal=None, *, noise=None, overdraft=0.,
         boost=.5, fixed_repulsor=.5, dynamic_repulsor=.75):
    """Mutates balances once; accepts an already prepared eligible distribution.

    The reference adjusts every token before sampling, rather than selecting
    per-bin winners as the production kernel does.
    """
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
    proposal_bin = (int(bins[proposal]) if proposal is not None and
                    0 <= proposal < len(probs) and eligible[proposal] else None)
    if proposal_bin is not None and balances[proposal_bin] + overdraft > 0:
        selected, token, path = proposal_bin, proposal, 'proposal'
    else:
        if noise is None:
            _, noise = element_noise(seed, len(probs))
        weights, path = fallback_weights(
            balances, present, proposal_bin, proposal_given=proposal is not None,
            overdraft=overdraft, boost=boost, fixed_repulsor=fixed_repulsor,
            dynamic_repulsor=dynamic_repulsor)
        with np.errstate(divide='ignore', invalid='ignore'):
            objective = np.where(eligible & (weights[bins] > 0),
                                 np.log(probs) + noise + np.log(weights[bins]) - np.log(mass[bins]),
                                 -np.inf)
        token = int(np.argmax(objective))
        selected = int(bins[token])
    if np.count_nonzero(present) == 1:
        balances[:] = before
    else:
        balances[selected] -= 1
    return int(token), dict(path=path, bin=int(selected), mass=mass)
