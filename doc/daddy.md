# DADDy sampling

DADDy operates on the eligible token distribution after the configured penalties,
temperature, filters, and constraints. The remaining token probabilities are
normalized to sum to one. Call this distribution `q_t(x)` at step `t`.

DADDy uses those probabilities in two different ways:

- **Bin assignment:** divide each token's probability by the highest eligible
  token probability: `q_t(x) / max(q_t)`.
- **Capacity earned:** add the token's actual normalized probability, `q_t(x)`,
  to its bin's credit.

These are the eight bins used by the implementation:

| Bin | Probability relative to the most likely token |
|---|---|
| 0 | 0–3% |
| 1 | 3–8% |
| 2 | 8–15% |
| 3 | 15–23% |
| 4 | 23–32% |
| 5 | 32–45% |
| 6 | 45–65% |
| 7 | 65–100% |

Internal boundaries belong to the higher bin: exactly 3% belongs to bin 1,
exactly 8% to bin 2, and so on. The most likely token belongs to bin 7.
Zero-probability tokens are ineligible and contribute nothing.

For example, if the most likely token has probability **20%**, a token with
probability **1%** has relative probability **5%**. It belongs to bin 1 and
contributes **0.01 credit**, not 0.05.

Bin membership is recomputed at every step. These ranges are neither token
ranks nor fixed token counts. A bin's width does not determine its capacity:
the total actual probability of its eligible tokens does. APIs often expose
token likelihoods through a field called "logprobs"; the accounting here
uses normalized probabilities, not their logarithms.

## One signed balance per bin

***D**istribution **A**ccurate **D**ecoding **D**ecision polic****y***, or DADDy, keeps one credit
balance per bin during a generation. It updates that balance directly: add the
current probability mass, then subtract one if a token is selected from the bin.
It does not store separate cumulative probability totals or per-bin token counts.

1. Each step, every bin earns the total probability mass of its eligible tokens.
2. A populated bin is **open** when `credit + overdraft > 0`, and **closed**
   otherwise, including exact equality. A bin without eligible tokens is **unused**
   regardless of its retained credit. Overdraft defaults to zero.
3. Selecting one token spends **one full credit** from its bin.
4. A selection can overdraw a bin. Later probability contributions replenish it.

```python
for b in range(8):
    balance[b] += sum(q_t(x) for x in eligible_tokens if bin_of(x) == b)
# Choose one token, then:
balance[selected_bin] -= 1.0
```

**An open bin need not have room for a whole token.** If its credit is 0.05,
it can receive a token and end with −0.95 credit. This debt closes it until
enough credit returns; it does not invalidate the token just selected.

The equivalent "fullness" convention reverses the signs: subtract probability
mass, allow selection when fullness is negative, and add one on selection.
The implementation uses positive credit throughout. Reversing every sign and
comparison gives the same decisions and numerical precision; either convention
can keep the net balance directly without subtracting growing totals.

## Using open bins for speculative decoding

After credit is added, an eligible speculative proposal is selected directly if
its bin is open. This gives proposals a **larger acceptance target** than
matching an independently sampled token in this generator: any eligible token
in an open bin can be selected as the proposal.

If no proposal is supplied, or the proposal is ineligible or its bin is closed,
DADDy falls back to a random bin selection weighted by available credit, then a
probability-weighted token selection inside that bin:

```text
add the current eligible probability masses to the bin scores
available_credit = score + overdraft
if the proposal is eligible and its bin has positive available_credit:
    select the proposal
else:
    weights = positive available_credit of populated bins
    if there is a proposal and at least one open bin:
        shares = weights / sum(weights)
        weights = shares + 0.5 * min(shares, 0.125)
        apply the fixed and dynamic repulsor multipliers
        if every weight was suppressed:
            restore the original available-credit weights
    select a bin using these weights, and a token proportionally within it
subtract one from the selected bin's score
```

The small-share curve gives tiny shares a 50% boost, capped at 0.0625 added
weight per bin. It depends on selection share, not bin index or token probability.
The fixed repulsor is centered on the third bin (index 2). The dynamic repulsor is
centered on the eligible proposal's bin and is omitted for invalid or filtered-out
proposals. Their multipliers combine by multiplication:

| Distance from center | Fixed weight reduction | Dynamic weight reduction |
|---|---:|---:|
| Same bin | 50% | 75% |
| One bin away | 25% | 37.5% |
| Two bins away | 12.5% | 18.75% |
| Farther away | None | None |

The curve and both repulsors apply **only when a speculative token is supplied**,
and only on fallback. An ineligible proposal still enables the curve and fixed
repulsor. Three internal strengths independently disable these biases at zero;
repulsors may reach 100% internally. These are starting heuristics for preserving
credit near regions useful to speculation, not guarantees of faster decoding.
They do not change the scores or direct proposal acceptance.

**All-closed exception:** if no populated bin has positive available credit,
select the populated bin with the greatest score. Tied bins have equal selection
weight, with proportional token selection inside each. This can happen when
credit is stranded in unused bins, and may spend below the ordinary opening
threshold. The curve and repulsors do not apply to this exception.

Without speculation and with zero overdraft, the first fallback from zero scores
matches the eligible categorical distribution. Later draws remain stateful.

### Overdraft

`DADDySampler(sampler, overdraft=0.5)` allows fractional overdraft. The default is
`0.0`; finite values from **0.0 through 5.0 inclusive** are accepted. Chat and the
speculative-decoding evaluator accept `--daddy --daddy-overdraft 0.5` (also spelled
`-daddy-o`). A nonzero CLI overdraft requires `--daddy`.

With overdraft 0.5, a bin is open above score -0.5; exactly -0.5 is closed.
Available credit also supplies fallback weights. Overdraft is not earned credit
and is never added to the stored scores. Selecting a token still spends one, so
the opening threshold is not a hard debt limit after selection.

This is a potentially **very destructive quality/speed tradeoff**, with values
above one especially experimental. It can accept proposals that would otherwise
be rejected and flatten fallback bin preferences, including at startup. Higher
acceptance does not guarantee higher throughput or unchanged output quality.

### What bin balancing does and does not preserve

Balances track accumulated expected bin occupancy minus actual selections
along the distributions encountered during this generation. They do not
guarantee preservation of the original token distribution. Proposal preference
can favor particular tokens within a bin even when the combined bin budget is
well balanced. For example, tokens with probabilities 2% and 3% share bin 1 when
the maximum is 50%; a shared 5% budget does not enforce their individual 2:3
selection ratio. DADDy is a heuristic sampling policy, not a claim of lossless
speculative sampling relative to the wrapped categorical distribution.

Rare tokens can still occur consecutively. Their combined bin mass may be
substantial, several bins may be open, and the distribution changes with each
new context. Splitting a range into more bins creates more independent balances
and can allow more early selections across those bins. The boundaries above
are a design choice, not a demonstrated optimum.

Filtering changes the eligible tokens and their normalized masses before these
updates. Empty bins earn no new credit and cannot be selected, but retain past
balances. Increasing min-P can therefore remove low-probability bins from
fallback selection. Existing balances are not rescaled or reset by
a change in the eligible distribution.

After a normal committed selection, the sum of balances stays approximately
zero, subject to reduction rounding. Persistent balances use double precision;
vocabulary mass accumulation uses float precision and a deterministic reduction
order. A singleton eligible distribution adds and spends one unit in bin 7,
leaving balances unchanged. Greedy sampling keeps the existing first-index tie
rule. Forced output injection also acts as a singleton and leaves balances
unchanged. An entirely empty eligible set retains the existing fused sampler's
token-zero convention and does not update balances.

## Usage and compatibility

```python
from exllamav3 import ComboSampler, DADDySampler, Job

sampler = DADDySampler(ComboSampler(temperature=0.8, min_p=0.08))
job = Job(input_ids=prompt_ids, sampler=sampler, max_new_tokens=256, seed=1234)
generator.enqueue(job)
```

In `chat.py`, add `--daddy` with your usual sampling settings. It works with or
without speculative decoding:

```bash
python examples/chat.py -m /path/to/model -mode chatml --daddy -tps
```

Choose the prompt mode appropriate for your model. Omitting `--daddy` keeps the
existing sampler. The legacy `--speculative_fallback_modulo 4` and
`--speculative-fallback-modulo 4` spellings remain accepted for existing launch
scripts as no-ops; other values are rejected. Directional selection is removed,
so seeded DADDy outputs intentionally change.

Rebuild the native extension after installing this change; an older precompiled
extension may lack `daddy_sampler` or have an incompatible native API version.
The wrapper checks API version 2 and reports a rebuild instruction for stale binaries.
The existing fused sampler must be enabled (`EXL3_FUSED_SAMPLER` must not be `0`).
The wrapped sampler must have a fused-compatible terminal pipeline. This includes
standard categorical, greedy, min-P, top-K, top-P, and ComboSampler presets, with
supported leading penalties and biases. Unsupported pipelines, such as active
Adaptive-P or XTC, raise `ValueError` at construction. With no wrapped sampler
supplied, DADDy uses `DefaultSampler`.

## State and generation integration

The generator passes the speculative token into DADDy. Every selection path returns one token. The generator's existing equality check then continues speculation on a match, or commits the returned token and discards the remaining speculative tail on a mismatch. A drawn token that happens to match the proposal also continues speculation. No separate acceptance flag or second sampling call is used, and discarded positions never update DADDy state.

Each `Job` owns independent state even when jobs share the sampler configuration. State persists across speculative rounds and requeues. Banned-string rewinds restore balances and internal decision counters to their checkpoint. Replacing a job's sampler starts fresh state and releases any held checkpoint belonging to the old sampler.

For direct calls outside a `Job`:

```python
sampler.reset()  # begin a new generation, with one state stream per batch row
output = sampler.forward(logits, rand_u32=seed, speculative_token=proposal)
```

`proposal` is an optional contiguous `torch.long` tensor with one ID per flattened batch row, on the logits device. The returned tensor is a reusable buffer, overwritten by the next call. Clone it if it must survive that call. Reset before changing batch size or device. Explicit independent contexts can be created with `sampler.new_generation()` and passed as `generation_state=...`.

The fused kernel accumulates eight masses and per-bin Gumbel winners without allocating vocabulary-sized probability or bin tensors. The small final reduction updates state and chooses the output. It reuses each
bin's existing Gumbel winner: add the logarithm of the desired bin weight and
subtract the logarithm of its current unnormalized mass, then choose the largest
adjusted winner. This fuses bin and token selection without extra random draws,
vocabulary passes, kernel launches, or vocabulary-sized buffers. Final credit
and weight arithmetic uses double precision; exact adjusted-score ties choose
the lowest token index. Bins whose float mass underflows to zero are excluded
from the logarithmic fallback draw. Sampling modes without an existing maximum pass need one to locate the relative-probability bins. Sampling, partials, state, and output remain on the GPU. The generator uploads CPU draft windows once per round using a reusable buffer. The default sampler retains its original GPU specialization and launch count.

Diagnostics require an explicit readback, outside the hot path:

```python
stats = job.daddy_state.stats()[0]
# balances, direct_proposals, proposal_draws, all_full_fallbacks
```

`direct_proposals` counts eligible proposals selected directly. `proposal_draws`
counts proposal steps requiring a fallback draw, including draws that match the
proposal. `all_full_fallbacks` counts steps where every currently eligible bin
is closed at the configured overdraft threshold. These counters record internal decisions rather than
bin occupancies, which can be computed externally from the final distribution
and selected tokens. Forced tokens leave both balances and counters unchanged.
Checkpoint restoration also restores the counters.

## Validation

Run the CPU lifecycle/reference tests and GPU/model tests from the repository root:

```bash
python -m pytest tests/generator/sampler/test_daddy.py
python -m pytest tests/kernels/sampling/test_daddy_sampler.py tests/kernels/sampling/test_fused_sampler.py
python -m pytest tests/e2e/test_daddy_generation.py --slow --model-root /path/to/models
```

GPU tests use the test suite's configurable device fixture. Model tests resolve
the `dense` role through the model registry; local paths belong in untracked
registry overrides or command-line configuration. Missing GPUs or models skip
the corresponding tests. No throughput or quality improvement is claimed here.
