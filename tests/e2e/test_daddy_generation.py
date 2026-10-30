"""Model integration: a single DADDy call per committed step, with n-gram drafting."""
from unittest.mock import patch
import pytest
import torch
from exllamav3 import Job, DADDySampler, ComboSampler
from testlib.e2e import load_model

pytestmark = [pytest.mark.model('dense'), pytest.mark.slow]


@pytest.mark.parametrize('overdraft', [0., .5])
@pytest.mark.parametrize('draft', [False, True])
def test_one_call_and_budget_per_committed_position(model_registry, device, draft, overdraft):
    with load_model(model_registry.get('dense').path, device) as lm:
        gen = lm.generator(ngram_match_min=2 if draft else 0, num_draft_tokens=4,
                           record_draft_stats=True)
        sampler = DADDySampler(ComboSampler(top_k=32, temperature=.8), overdraft=overdraft)
        ids = lm.tokenizer.encode('One two three four. ' * 50, add_bos=True)
        job = Job(ids, max_new_tokens=48, stop_conditions=[], sampler=sampler, seed=123)
        with patch.object(sampler, 'forward', wraps=sampler.forward) as calls:
            gen.enqueue(job)
            while gen.num_remaining_jobs():
                gen.iterate()
            assert calls.call_count == job.new_tokens == 48
        stats = job.daddy_state.stats()[0]
        assert abs(sum(stats['balances'])) < 1e-10
        if draft:
            assert job.accepted_draft_tokens + job.rejected_draft_tokens > 0
            assert any(call.kwargs['speculative_token'] is not None for call in calls.call_args_list)
        gen.close()
