# DADDy sampling

**Distribution Accurate Decoding Decision policy** is an optional, stateful sampler for ordinary and speculative generation. It balances coarse probability bins over the distributions encountered during one generation. It is a heuristic: bin balancing and proposal preference do not guarantee preservation of the target token distribution.

```python
from exllamav3 import ComboSampler, DADDySampler, Job

sampler = DADDySampler(
    ComboSampler(temperature=0.8, min_p=0.08),
)
job = Job(input_ids=prompt_ids, sampler=sampler, max_new_tokens=256, seed=1234)
generator.enqueue(job)
```

In `chat.py`, add `--daddy` to enable the policy with your usual temperature,
min-P, top-K, and top-P settings. It works with or without speculative decoding:

```bash
python examples/chat.py -m /path/to/model -mode chatml --daddy -tps
```

Choose the prompt mode appropriate for your model. Omitting `--daddy` keeps the
existing sampler. The fallback policy is fixed at modulo 4. The old
`--speculative_fallback_modulo 4` and `--speculative-fallback-modulo 4` spellings
remain accepted for existing launch scripts; other values are rejected.

Rebuild the native extension after installing this change; an older precompiled extension may lack `daddy_sampler` or use an incompatible diagnostic layout. The existing fused sampler must be enabled (`EXL3_FUSED_SAMPLER` must not be `0`). The wrapped sampler must have a fused-compatible terminal pipeline. This includes the standard categorical, greedy, min-P, top-K, top-P, and ComboSampler presets, with supported leading penalties and biases. Unsupported pipelines, such as active Adaptive-P or XTC, raise `ValueError` at construction. With no wrapped sampler supplied, DADDy uses `DefaultSampler`.

## Policy

Let `q_t(x)` be the final normalized eligible distribution after penalties, temperature, sampling truncation, and constraints. DADDy bins each eligible token by `q_t(x) / max(q_t)`. The bin boundaries, in percent, are:

```
0, 3, 8, 15, 23, 32, 45, 65, 100
```

These are relative-likelihood ranges, not equal-width bins, token ranks, or
fixed token counts. Bin membership is recomputed from each step's final eligible
distribution. For example, if the most likely token has probability 0.20, a
0.01-probability token has relative probability 5% and belongs to bin 1 (3–8%).
Each bin's credit uses its total actual probability mass, not its width or the
number of tokens it contains. Empty bins receive no new credit.

An internal boundary belongs to the bin on its right; the last bin includes 100%. Bin identity always belongs to the sampler state, including when classifying a supplied proposal. Each bin balance represents accumulated expected occupancy minus committed occupancy:

```
balance[b] += sum(q_t(x) for eligible x in bin b)
```

This update occurs before selection. If a supplied speculative token is eligible and its bin has positive balance, DADDy selects it. Otherwise, DADDy searches for a positive-balance bin containing eligible tokens, then makes a probability-weighted draw within that bin. Direction is fixed: high-first when the step seed modulo 4 is zero, and low-first otherwise (implemented as `seed & 3`). This makes three out of four fallback draws low-first. The intent is to skew fallback selections toward lower-probability bins, preserving credit in higher-probability bins so later draft proposals can be accepted and speculative runs can continue longer. This is a heuristic, not a guarantee of longer runs.

If every currently eligible bin is closed, select the eligible bin with the largest balance. Equal balances use the same direction to break ties. This situation can occur when credit accumulated in bins absent from the current distribution.

Finally, subtract one from the selected token's bin. Temporarily negative balances are expected. The sum of balances stays approximately zero, subject to reduction rounding. Double precision is used for persistent balances; vocabulary mass accumulation uses float precision and a deterministic reduction order.

A singleton eligible distribution adds and spends one unit in bin 7, leaving balances unchanged. Greedy sampling keeps the existing first-index tie rule. Forced output injection also acts as a singleton distribution and leaves balances unchanged. Existing generation masks, including packed bitmasks, restrict the distribution before the bin update. An entirely empty eligible set retains the existing fused sampler's token-zero convention and does not update DADDy's balances.

## One call, one buffered token

The generator passes the speculative token into DADDy. Every selection path returns one token. The generator's existing equality check then continues speculation on a match, or commits the returned token and discards the remaining speculative tail on a mismatch. A drawn token that happens to match the proposal also continues speculation. No separate acceptance flag or second sampling call is used, and discarded positions never update DADDy state.

Each `Job` owns independent state even when jobs share the sampler configuration. State persists across speculative rounds and requeues. Banned-string rewinds restore balances and internal decision counters to their checkpoint. Replacing a job's sampler starts fresh state and releases any held checkpoint belonging to the old sampler.

For direct calls outside a `Job`:

```python
sampler.reset()  # begin a new generation, with one state stream per batch row
output = sampler.forward(logits, rand_u32=seed, speculative_token=proposal)
```

`proposal` is an optional contiguous `torch.long` tensor with one ID per flattened batch row, on the logits device. The returned tensor is a reusable buffer, overwritten by the next call. Clone it if it must survive that call. Reset before changing batch size or device. Explicit independent contexts can be created with `sampler.new_generation()` and passed as `generation_state=...`.

The fused kernel accumulates eight masses and per-bin Gumbel winners without allocating vocabulary-sized probability or bin tensors. The small final reduction updates state and chooses the output. Sampling modes without an existing maximum pass need one to locate the relative-probability bins. Sampling, partials, state, and output remain on the GPU. The generator uploads CPU draft windows once per round using a reusable buffer. The default sampler retains its original GPU specialization and launch count.

Diagnostics require an explicit readback, outside the hot path:

```python
stats = job.daddy_state.stats()[0]
# balances, direct_proposals, proposal_draws, all_full_fallbacks
```

`direct_proposals` counts eligible proposals selected directly. `proposal_draws`
counts proposal steps requiring a fallback draw, including draws that match the
proposal. `all_full_fallbacks` counts steps where every currently eligible bin
has nonpositive balance. These counters record internal decisions rather than
bin occupancies, which can be computed externally from the final distribution
and selected tokens. Forced tokens leave both balances and counters unchanged.
Checkpoint restoration also restores the counters.

## Validation

Run the CPU lifecycle/reference tests and the GPU/model tests from the repository root:

```bash
python -m pytest tests/generator/sampler/test_daddy.py
python -m pytest tests/kernels/sampling/test_daddy_sampler.py tests/kernels/sampling/test_fused_sampler.py
python -m pytest tests/e2e/test_daddy_generation.py --slow --model-root /path/to/models
```

GPU tests use the test suite's configurable device fixture. Model tests resolve
the `dense` role through the model registry; local paths belong in untracked
registry overrides or command-line configuration. Missing GPUs or models skip
the corresponding tests. No throughput or quality improvement is claimed here.
