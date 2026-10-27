"""DADDy fused kernels against the CPU policy oracle and Philox noise stream."""
import math
import numpy as np
import pytest
import torch
from exllamav3 import DADDySampler, CategoricalSampler, ComboSampler, GreedySampler
from exllamav3.ext import exllamav3_ext as ext
from testlib.daddy import distribution, step
from testlib.sampling import element_noise


def invoke(logits, balances, counts, *, seed=0, modulo=4, proposal=None, mode=1,
           inv_temp=1., minp_log=0., filters=0, top_k=0, top_p=1., inv_temp_filter=1.,
           mask=None, bits=None, size=None, buffers=None):
    rows, dim = logits.shape
    if buffers is None:
        buffers = (
            torch.empty((rows, 1), dtype=torch.long, device=logits.device),
            torch.empty(rows * ext.FUSED_SAMPLER_MAX_BLOCKS * 3, device=logits.device),
            torch.empty(rows * ext.FUSED_SAMPLER_HIST_STRIDE, dtype=torch.uint8, device=logits.device),
            torch.empty(rows * ext.FUSED_SAMPLER_MAX_BLOCKS * ext.DADDY_SAMPLER_PARTIAL_FLOATS, device=logits.device),
        )
    out, ws, hist, dws = buffers
    ext.daddy_sampler(logits, mask, bits, out, ws, size or dim, inv_temp, minp_log,
                      seed, mode, filters, top_k, top_p, inv_temp_filter,
                      hist if mode == 3 else None, dws, balances, counts, proposal, modulo)
    return out, buffers


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('case', [
    dict(mode=1), dict(mode=2, minp_log=math.log(.08), inv_temp=1.25),
    dict(mode=3, filters=3, top_k=37, top_p=.83, inv_temp=.8),
    dict(mode=3, filters=7, top_k=51, top_p=.91, minp_log=math.log(.05), inv_temp=1.25, inv_temp_filter=1.25),
])
def test_multiround_reference(device, dtype, case):
    rows, dim = 3, 4097
    rng = np.random.default_rng(42)
    balances = torch.zeros(rows, 8, dtype=torch.double, device=device)
    counts = torch.zeros(rows, ext.DADDY_SAMPLER_COUNTS, dtype=torch.long, device=device)
    expected_balance = np.zeros((rows, 8))
    expected_counts = np.zeros((rows, 3), dtype=np.int64)
    buffers = None
    for seed in range(9):
        x = rng.normal(0, 1.7, (rows, dim)).astype(np.float32)
        # A coarse exactly representable grid avoids histogram boundary ambiguity.
        x = np.round(x * 64) / 64
        logits = torch.tensor(x, dtype=dtype, device=device)
        proposal = None if seed % 3 == 0 else torch.tensor(
            rng.integers(0, dim, rows), dtype=torch.long, device=device)
        out, buffers = invoke(logits, balances, counts, seed=seed,
                              proposal=proposal, buffers=buffers, **case)
        _, noise = element_noise(seed, rows * dim)
        expected = []
        for row in range(rows):
            probs = distribution(logits[row].float().cpu().numpy(), **case)
            token, info = step(probs, expected_balance[row], seed,
                               None if proposal is None else int(proposal[row]),
                               noise=noise.reshape(rows, dim)[row])
            expected.append(token)
            expected_counts[row, 0] += int(info['path'] == 'proposal')
            expected_counts[row, 1] += int(proposal is not None and info['path'] != 'proposal')
            expected_counts[row, 2] += int(info['path'] == 'all_full')
        assert out.flatten().tolist() == expected
        np.testing.assert_allclose(balances.cpu(), expected_balance, rtol=0, atol=2e-5)
        np.testing.assert_array_equal(counts.cpu(), expected_counts)


def test_exact_boundaries(device):
    # The supplied fp32 log thresholds belong to the bin on their right.
    x = torch.tensor([[math.log(p) for p in [.03, .08, .15, .23, .32, .45, .65, 1.]]], device=device)
    for candidate in range(8):
        balance = torch.zeros(1, 8, dtype=torch.double, device=device)
        counts = torch.zeros(1, ext.DADDY_SAMPLER_COUNTS, dtype=torch.long, device=device)
        out, _ = invoke(x, balance, counts, proposal=torch.tensor([candidate], device=device))
        assert out.item() == candidate
        assert counts[0, 0] == 1
        probs = distribution(x[0].cpu().numpy(), mode=1)
        bins = np.minimum(np.arange(8) + 1, 7)
        expected_balance = np.bincount(bins, weights=probs, minlength=8)
        expected_balance[min(candidate + 1, 7)] -= 1
        np.testing.assert_allclose(balance.cpu()[0], expected_balance, atol=2e-6)


def test_all_full_and_fallback_matching_proposal(device):
    probs = torch.tensor([[1/128] * 64 + [.5]], device=device)
    balance = torch.zeros(1, 8, dtype=torch.double, device=device)
    balance[0, [0, 3, 7]] = torch.tensor([-.8, 1.4, -.6], dtype=torch.double, device=device)
    counts = torch.zeros(1, ext.DADDY_SAMPLER_COUNTS, dtype=torch.long, device=device)
    out, _ = invoke(probs.log(), balance, counts,
                    proposal=torch.tensor([64], device=device))
    assert out.item() == 64  # overfilled proposal's bin wins the all-full fallback
    assert counts[0, 2] == 1 and counts[0, 1] == 1
    assert counts[0, 0] == 0  # diagnostics do not control acceptance: token equality does


@pytest.mark.parametrize('mask_kind', ['half', 'bits'])
def test_constraints_and_vocab_bound(device, mask_kind):
    logits = torch.tensor([[8., 5., 3., 1., 10.]], device=device)
    sampler = DADDySampler(CategoricalSampler())
    if mask_kind == 'half':
        mask = torch.tensor([[-float('inf'), 0., 0., -float('inf')]], dtype=torch.half, device=device)
    else:
        mask = torch.tensor([[0b110]], dtype=torch.int32, device=device)
    for seed in range(12):
        out = sampler.forward(logits, rand_u32=seed, logit_mask=mask,
                              speculative_token=torch.tensor([0], device=device))
        assert out.item() in [1, 2]
    assert sampler.state.stats()[0]['proposal_draws'] == 12
    assert abs(sampler.state.balances.sum().item()) < 1e-12


@pytest.mark.parametrize('greedy', [False, True])
def test_singleton_and_greedy_ties(device, greedy):
    sampler = DADDySampler(GreedySampler() if greedy else ComboSampler(min_p=.1))
    logits = torch.tensor([[0., 0. if greedy else -30., -30.]], device=device)
    sampler.state.prepare(device, 1)
    sampler.state.balances[0] = torch.tensor([.2, -.4, .7, -.3, 0., 0., .1, -.3], device=device)
    before = sampler.state.balances.clone()
    for seed in range(20):
        assert sampler.forward(logits, rand_u32=seed, speculative_token=torch.tensor([1], device=device)).item() == 0
    assert torch.equal(sampler.state.balances, before)
    assert sampler.state.stats()[0]['proposal_draws'] == 20


def test_topp_partial_tie_mass_and_temperature(device):
    sampler = DADDySampler(ComboSampler(top_p=.7, temperature=.8, temp_last=True))
    x = torch.tensor([[0., -1., -1., -2.]], device=device)
    token = sampler.forward(x, rand_u32=42).item()
    probs = distribution(x[0].cpu(), mode=3, filters=2, top_p=.7, inv_temp=1.25)
    relative = probs / probs.max()
    bins = np.searchsorted([.03, .08, .15, .23, .32, .45, .65], relative, side='right')
    mass = np.bincount(bins, weights=probs, minlength=8)
    mass[bins[token]] -= 1
    np.testing.assert_allclose(sampler.state.balances.cpu()[0], mass, atol=2e-6)


def test_replay_same_seeds_and_reuse_buffers(device):
    sampler = DADDySampler(CategoricalSampler())
    x = torch.tensor([[0., -1., -2., -3., -4.]], device=device)
    def run():
        tokens, ptr = [], None
        for seed in [7, 29, 31, 4, 99, 432]:
            result = sampler.forward(x, rand_u32=seed)
            tokens.append(result.item())
            if ptr is None:
                ptr = result.data_ptr()
            assert result.data_ptr() == ptr
        return tokens, sampler.state.balances.clone(), sampler.state.counts.clone()
    a = run()
    sampler.reset()
    b = run()
    assert a[0] == b[0] and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])


def test_single_bin_is_exactly_neutral_after_accumulation(device):
    sampler = DADDySampler(CategoricalSampler())
    sampler.state.prepare(device, 1)
    sampler.state.balances[0, 7] = -.3  # non-dyadic fp64; add/subtract one can round
    before = sampler.state.balances.clone()
    for seed in range(10):
        sampler.forward(torch.zeros((1, 3), device=device), rand_u32=seed)
    assert torch.equal(sampler.state.balances, before)


def test_topp_maximum_tie_bias_defines_pmax(device):
    sampler = DADDySampler(ComboSampler(top_p=.4))
    token = sampler.forward(torch.zeros(1, 4, device=device), rand_u32=0,
                            speculative_token=torch.tensor([2], device=device))
    assert token.item() == 2
    stats = sampler.state.stats()[0]
    assert stats['direct_proposals'] == 1
    assert stats['balances'] == [0.] * 8


@pytest.mark.parametrize('modulo', [0, 1, 2, 8, 16])
def test_native_modulo_is_fixed(device, modulo):
    logits = torch.zeros((1, 3), device=device)
    balances = torch.zeros((1, 8), dtype=torch.double, device=device)
    counts = torch.zeros((1, ext.DADDY_SAMPLER_COUNTS), dtype=torch.long, device=device)
    with pytest.raises(RuntimeError, match='fixed at 4'):
        invoke(logits, balances, counts, modulo=modulo)
    assert balances.count_nonzero() == 0 and counts.count_nonzero() == 0
