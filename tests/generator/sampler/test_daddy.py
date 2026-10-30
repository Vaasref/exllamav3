"""CPU policy oracle and sampler/job lifecycle contracts; no model or GPU needed."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from exllamav3 import Job, DADDySampler, CategoricalSampler, CustomSampler, SS_Sample_mn
from exllamav3.generator.sampler.daddy import DADDyState
from exllamav3.ext import exllamav3_ext as ext
from testlib.daddy import BOUNDARIES, distribution, step, fallback_weights
from testlib.sampling import element_noise

pytestmark = pytest.mark.nogpu


@pytest.fixture
def cpu_kernel(monkeypatch):
    calls = []
    monkeypatch.setattr(ext, 'DADDY_SAMPLER_COUNTS', 3, raising=False)
    monkeypatch.setattr(ext, 'DADDY_SAMPLER_API_VERSION', 2, raising=False)
    monkeypatch.setattr(ext, 'DADDY_SAMPLER_PARTIAL_FLOATS', 26, raising=False)
    def run(logits, mask, bits, out, ws, size, inv_temp, minp_log, seed, mode,
            filters, top_k, top_p, inv_temp_filter, hist, dws, balances, counts, proposal, overdraft, boost, fixed_repulsor, dynamic_repulsor):
        calls.append(None if proposal is None else proposal.clone())
        for row, tensor in enumerate(logits):
            x = tensor.double().numpy().copy()
            x[size:] = -np.inf
            if mask is not None:
                x[:size] += mask[min(row, len(mask)-1), :size].numpy()
            if bits is not None:
                words = bits[min(row, len(bits)-1)].numpy()
                for i in range(size):
                    if not (int(words[i//32]) >> (i % 32)) & 1:
                        x[i] = -np.inf
            probs = distribution(x, mode, inv_temp=inv_temp, minp_log=minp_log,
                                 filters=filters, top_k=top_k, top_p=top_p,
                                 inv_temp_filter=inv_temp_filter)
            _, noise = element_noise(seed, logits.numel())
            candidate = None if proposal is None else int(proposal[row])
            token, info = step(probs, balances[row].numpy(), seed, candidate,
                               noise=noise.reshape(logits.shape)[row], overdraft=overdraft, boost=boost,
                               fixed_repulsor=fixed_repulsor, dynamic_repulsor=dynamic_repulsor)
            out[row, 0] = token
            if info['bin'] is not None:
                counts[row, 0] += int(info['path'] == 'proposal')
                counts[row, 1] += int(candidate is not None and info['path'] != 'proposal')
                counts[row, 2] += int(info['path'] == 'all_full')
    monkeypatch.setattr(ext, 'daddy_sampler', run, raising=False)
    monkeypatch.setattr(ext, 'generator_stage_decode', lambda *args: None, raising=False)
    return calls


def test_boundary_convention():
    p = np.r_[BOUNDARIES, 1.]
    # Equal relative thresholds belong to the higher bin.
    assert np.searchsorted(BOUNDARIES, p / p.max(), side='right').tolist() == list(range(1, 8)) + [7]


def test_first_step_matches_categorical():
    p = np.array([.02, .08, .9])
    for seed in range(16):
        balance = np.zeros(8)
        token, info = step(p, balance, seed)
        _, noise = element_noise(seed, len(p))
        assert token == np.argmax(np.log(p) + noise)
        assert abs(balance.sum()) < 1e-14
        assert info['path'] == 'draw'


def test_singleton_never_changes_balances():
    original = np.array([.4, -.3, .2, 0., -.1, .5, -.2, -.5])
    balance = original.copy()
    for seed in range(100):
        token, _ = step([0., 1., 0.], balance, seed, proposal=1)
        assert token == 1
    np.testing.assert_array_equal(balance, original)


def test_changing_distribution_all_full_and_equal_bin_ties():
    # Empty intermediate bins retain positive credit; currently present bins have debt.
    p = np.r_[np.full(64, 1/128), .5]
    for seed in [0, 1]:
        b = np.zeros(8)
        b[[0, 3, 7]] = [-.8, 1.4, -.6]
        token, info = step(p, b, seed)
        assert info['path'] == 'all_full'
        assert token == 64  # -.1 in bin 7 is larger than -.3 in bin 0
    # Equal-score bins receive equal total weight, despite very different token counts.
    selected = []
    for seed in range(1000):
        b = np.zeros(8)
        b[[0, 3, 7]] = [-.75, 1.5, -.75]
        _, info = step(p, b, seed)
        selected.append(info['bin'])
    assert .45 < np.mean(np.array(selected) == 7) < .55


def test_proposal_is_taken_by_bin_without_reference_token_equality():
    b = np.zeros(8)
    token, info = step([.45, .55], b, 0, proposal=0, noise=np.array([-100., 100.]))
    assert token == 0 and info['path'] == 'proposal'
    for proposal in [-1, 2]:
        token, info = step([0., 1.], np.zeros(8), 0, proposal=proposal)
        assert token == 1 and info['path'] == 'draw'
    token, info = step([0., 1.], np.zeros(8), 0, proposal=0)
    assert token == 1 and info['path'] == 'draw'


def test_weighted_draw_inside_bin():
    p = np.array([.44, .56])  # both in bin 7
    balance = np.zeros(8)
    tokens = [step(p, balance, seed)[0] for seed in range(2000)]
    assert abs(np.mean(tokens) - .56) < .04


def test_modulo_is_not_configurable():
    with pytest.raises(TypeError, match='speculative_fallback_modulo'):
        DADDySampler(speculative_fallback_modulo=4)


def test_unsupported_stack():
    with pytest.raises(ValueError, match='fused-compatible'):
        DADDySampler(CustomSampler([SS_Sample_mn()]))


def test_wrapper_buffer_reset_and_determinism(cpu_kernel):
    sampler = DADDySampler(CategoricalSampler())
    logits = torch.log(torch.tensor([[.02, .08, .9]]))
    first = sampler.forward(logits, rand_u32=1)
    address = first.data_ptr()
    balances = sampler.state.balances.clone()
    token = first.clone()
    sampler.reset()
    again = sampler.forward(logits, rand_u32=1)
    assert torch.equal(again, token)
    assert torch.equal(sampler.state.balances, balances)
    again_address = again.data_ptr()
    sampler.forward(logits, rand_u32=2)
    assert sampler.state.output.data_ptr() == again_address
    assert address != again_address
    assert len(cpu_kernel) == 3


def make_job(sampler):
    job = Job(torch.tensor([[1, 2, 3]]), max_new_tokens=100, sampler=sampler, seed=42)
    job.generator = SimpleNamespace(tokenizer=None)
    job.current_device_ids = None
    job.device_logit_mask = None
    return job


def test_job_once_per_step_proposal_and_shared_config(cpu_kernel):
    sampler = DADDySampler(CategoricalSampler())
    a, b = make_job(sampler), make_job(sampler)
    logits = torch.log(torch.tensor([[[.45, .55]]]))
    result = a.receive_logits(logits, speculative_token=torch.tensor([0]))
    assert result[0].item() == 0
    assert len(cpu_kernel) == 1 and cpu_kernel[0].item() == 0
    assert a.daddy_state is not b.daddy_state
    assert b.daddy_state.balances is None
    assert sampler.state.balances is None
    assert a.daddy_state.stats()[0]['direct_proposals'] == 1


def test_job_forced_output_is_singleton(cpu_kernel):
    job = make_job(DADDySampler(CategoricalSampler()))
    job.forced_ids = torch.tensor([[2]])
    logits = torch.zeros(1, 1, 3)
    with patch.object(job.sampler, 'forward', wraps=job.sampler.forward) as forward:
        token = job.receive_logits(logits)[0]
        assert forward.call_count == 1
    assert token.item() == 2 and job.forced_sample
    assert len(cpu_kernel) == 0
    assert torch.equal(job.daddy_state.balances, torch.zeros(1, 8, dtype=torch.double))
    assert job.daddy_state.counts.count_nonzero() == 0


def test_requeue_and_replacement(cpu_kernel):
    job = make_job(DADDySampler(CategoricalSampler()))
    job.receive_logits(torch.zeros(1, 1, 3))
    state = job.daddy_state
    job.cached_pages, job.cached_tokens = 0, 0
    job.new_tokens = 1
    job.sequences[0].sequence_ids.append(torch.tensor([[0]]))
    with patch.object(Job, 'prepare_for_queue'):
        job.prepare_for_requeue()
    assert job.daddy_state is state
    job.set_sampler(DADDySampler(CategoricalSampler()))
    assert job.daddy_state is not state and job.daddy_state.balances is None
    job.set_sampler(None)
    assert job.daddy_state is None


def test_checkpoint_snapshot_survives_future_steps(cpu_kernel):
    sampler = DADDySampler(CategoricalSampler())
    state = sampler.state
    state.prepare(torch.device('cpu'), 1)
    state.save_before_step()
    snapshot = state.checkpoint()
    for seed in range(4):
        sampler.forward(torch.zeros(1, 3), rand_u32=seed, speculative_token=torch.tensor([0]))
        state.save_before_step()
    assert state.counts.sum() > 0
    state.restore(snapshot)
    assert state.balances.count_nonzero() == 0 and state.counts.count_nonzero() == 0


def run_verification(job, logits, proposals):
    """Real verification loop with a stub model/cache and normal sampler calls."""
    from exllamav3 import Generator
    from exllamav3.constants import PAGE_SIZE
    gen = Generator.__new__(Generator)
    gen.model = SimpleNamespace(caps={}, forward=lambda **kw: logits, prefetch_tokens=lambda ids: None)
    gen.cache = None
    gen.tokenizer = None
    gen.active_jobs, gen.pending_jobs = [job], []
    gen.num_draft_tokens = len(proposals)
    gen.target_recurrent = gen.draft_recurrent = gen.dflash_draft = gen.mtp_draft = False
    gen.draft_model = gen.draft_calibrator = gen.recurrent_cache = None
    gen.record_draft_stats = True
    gen.daddy_draft_buffer = None
    gen.sparse_head = None
    def stage_arena(rows, width, with_positions):
        host = torch.zeros(rows * width + rows * (2 if with_positions else 1), dtype=torch.int32)
        return host, gen._arena_views(host, rows, width, with_positions), {}
    gen._stage_arena = stage_arena
    gen._upload_arena = lambda *args: None
    gen._staging = lambda name, rows, width=None, dtype=torch.int32: torch.zeros(
        (rows,) if width is None else (rows, width), dtype=dtype)
    seq = job.sequences[0]
    seq.kv_position = len(seq.sequence_ids) - 1
    seq.prefill_complete = True
    seq.block_index_tensor = torch.zeros((1, 1), dtype=torch.int32)
    seq.allocated_pages = [SimpleNamespace(kv_position=seq.kv_position, ref_count=1,
                                           sequence=torch.zeros((1, PAGE_SIZE), dtype=torch.long), can_revert=False)]
    job.generator = gen
    job.time_first_token = 1.
    job.prepare_logit_mask = job.prepare_sampling_past_ids = lambda: None
    committed = []
    def commit(logits, token, *args):
        token = token.clone()
        committed.append(token.item())
        job.new_tokens += 1
        seq.kv_position += 1
        seq.sequence_ids.append(token)
        return False, token, False
    job.receive_sample = commit
    gen.iterate_gen([], torch.tensor([proposals]))
    return committed


def test_verification_calls_once_then_commits_mismatch_and_discards_tail(cpu_kernel):
    job = make_job(DADDySampler(CategoricalSampler()))
    logits = torch.tensor([[[0., 0., 0.], [0., -float('inf'), -float('inf')],
                            [0., 0., 0.], [0., 0., 0.]]])
    committed = run_verification(job, logits, [1, 2, 0])
    assert committed == [1, 0]
    assert len(cpu_kernel) == 2
    assert [p.item() for p in cpu_kernel] == [1, 2]
    assert job.accepted_draft_tokens == 1 and job.rejected_draft_tokens == 2
    assert job.daddy_state.stats()[0]['direct_proposals'] + job.daddy_state.stats()[0]['proposal_draws'] == 2


def test_verification_accepts_fallback_that_matches_proposal(cpu_kernel):
    job = make_job(DADDySampler(CategoricalSampler()))
    job.daddy_state.prepare(torch.device('cpu'), 1)
    job.daddy_state.balances[0, [0, 3, 7]] = torch.tensor([-.8, 1.4, -.6], dtype=torch.double)
    logits = torch.log(torch.tensor([[[1/128] * 64 + [.5], [1/128] * 64 + [.5]]]))
    committed = run_verification(job, logits, [64])
    assert committed[0] == 64 and len(committed) == 2
    assert len(cpu_kernel) == 2 and cpu_kernel[1] is None
    assert job.accepted_draft_tokens == 1 and job.rejected_draft_tokens == 0
    assert job.daddy_state.counts[0, 0] == 0  # no direct acceptance, yet the matching draft continued


def test_mask_and_leading_bias_feed_final_distribution(cpu_kernel):
    from exllamav3 import ComboSampler
    sampler = DADDySampler(ComboSampler(logit_bias={0: -float('inf'), 1: 2.}))
    logits = torch.zeros((1, 4))
    mask = torch.tensor([[0., 0., -float('inf'), -float('inf')]], dtype=torch.half)
    token = sampler.forward(logits, rand_u32=12, logit_mask=mask,
                            speculative_token=torch.tensor([0]))
    assert token.item() == 1
    assert sampler.state.balances.count_nonzero() == 0


@pytest.mark.parametrize('outdated_layout', [False, True])
def test_missing_or_outdated_native_extension_has_actionable_error(monkeypatch, outdated_layout):
    if outdated_layout:
        monkeypatch.setattr(ext, 'daddy_sampler', lambda *args: None, raising=False)
        monkeypatch.setattr(ext, 'DADDY_SAMPLER_COUNTS', 13, raising=False)
    else:
        monkeypatch.delattr(ext, 'daddy_sampler', raising=False)
    with pytest.raises(RuntimeError, match='rebuilding the native'):
        DADDyState().prepare(torch.device('cpu'), 1)


@pytest.mark.parametrize('overdraft', [0., .125, .5, 1., 2., 5.])
def test_fractional_overdraft_configuration_and_reset(cpu_kernel, overdraft):
    sampler = DADDySampler(CategoricalSampler(), overdraft=overdraft)
    other = sampler.new_generation()
    assert other is not sampler.state
    assert other.overdraft == sampler.state.overdraft == overdraft
    sampler.reset()
    assert sampler.state.overdraft == overdraft
    # Proposal bin has -0.25 credit after earning the current mass.
    sampler.state.prepare(torch.device('cpu'), 1)
    sampler.state.balances[0, 0] = -.26
    sampler.state.balances[0, 7] = .26
    sampler.forward(torch.log(torch.tensor([[.01, .99]])), rand_u32=5,
                    speculative_token=torch.tensor([0]))
    assert sampler.state.stats()[0]['direct_proposals'] == int(overdraft > .25)


@pytest.mark.parametrize('overdraft', [-.1, 5.0001, float('inf'), -float('inf'), float('nan')])
def test_invalid_overdraft(overdraft):
    with pytest.raises(ValueError, match='overdraft must be finite'):
        DADDySampler(overdraft=overdraft)


def test_stale_api_version_requires_rebuild(monkeypatch):
    monkeypatch.setattr(ext, 'daddy_sampler', lambda *args: None, raising=False)
    monkeypatch.setattr(ext, 'DADDY_SAMPLER_COUNTS', 3, raising=False)
    monkeypatch.setattr(ext, 'DADDY_SAMPLER_API_VERSION', 1, raising=False)
    with pytest.raises(RuntimeError, match='rebuilding the native'):
        DADDyState().prepare(torch.device('cpu'), 1)


def test_exact_overdraft_threshold_is_closed():
    # Dyadic probabilities and balances make equality exact.
    p = [1/64, 63/64]
    balances = np.zeros(8)
    balances[[0, 7]] = [-.5 - 1/64, .5 + 1/64]
    _, info = step(p, balances, 0, proposal=0, overdraft=.5)
    assert info['path'] == 'draw' and info['bin'] == 7
    balances[[0, 7]] = [-.5 - 1/64 + 1/1024, .5 + 1/64 - 1/1024]
    _, info = step(p, balances, 0, proposal=0, overdraft=.5)
    assert info['path'] == 'proposal'


def test_bias_curve_bubbles_and_proposal_gating():
    credit = np.array([.01, .02, .03, .04, .10, .20, .25, .35])
    present = np.ones(8, dtype=bool)
    ordinary, _ = fallback_weights(credit, present)
    np.testing.assert_array_equal(ordinary, credit)
    boosted = credit + .5 * np.minimum(credit, .125)
    fixed = np.array([.875, .75, .5, .75, .875, 1., 1., 1.])
    dynamic = np.array([1., 1., 1., .8125, .625, .25, .625, .8125])
    actual, _ = fallback_weights(credit, present, 5, proposal_given=True)
    np.testing.assert_allclose(actual, boosted * fixed * dynamic)
    invalid, _ = fallback_weights(credit, present, proposal_given=True)
    np.testing.assert_allclose(invalid, boosted * fixed)
    disabled, _ = fallback_weights(credit, present, 5, proposal_given=True,
                                   boost=0., fixed_repulsor=0., dynamic_repulsor=0.)
    np.testing.assert_allclose(disabled, credit)


def test_full_suppression_reverts_to_original_credit():
    credit = np.zeros(8)
    credit[2] = .4
    weights, path = fallback_weights(credit, credit > 0, proposal_given=True, fixed_repulsor=1.)
    np.testing.assert_array_equal(weights, credit)
    assert path == 'draw'
    # Both populated bins can be suppressed by different repulsors.
    credit[7] = .6
    weights, _ = fallback_weights(credit, credit > 0, 7, proposal_given=True,
                                  fixed_repulsor=1., dynamic_repulsor=1.)
    np.testing.assert_array_equal(weights, credit)


def test_controlled_bin_weights_and_within_bin_distribution():
    p = np.array([.01, .02, .04, .93])  # first two share bin zero
    mass = np.zeros(8)
    mass[[0, 1, 7]] = [.03, .04, .93]
    target_credit = np.zeros(8)
    target_credit[[0, 1, 7]] = [.2, .3, .5]
    rng = np.random.default_rng(94)
    counts = np.zeros(4)
    for _ in range(12000):
        token, _ = step(p, target_credit - mass, 0, noise=rng.gumbel(size=4))
        counts[token] += 1
    np.testing.assert_allclose(counts / counts.sum(), [.2/3, .4/3, .3, .5], atol=.015)


@pytest.mark.parametrize('overdraft', [0., .5, 5.])
def test_changing_distribution_accounting(overdraft):
    rng = np.random.default_rng(19)
    balances = np.zeros(8)
    expected = np.zeros(8)
    selections = np.zeros(8)
    for turn in range(500):
        p = rng.dirichlet(np.full(64, .3))
        if turn % 7 == 0:
            p[p < .02] = 0
        _, info = step(p, balances, turn, proposal=turn % 64, overdraft=overdraft)
        expected += info['mass']
        selections[info['bin']] += 1
        np.testing.assert_allclose(balances, expected - selections, atol=2e-12)
        assert abs(balances.sum()) < 2e-12
