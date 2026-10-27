"""Distribution Accurate Decoding Decision policy (DADDy).

Configuration can be shared; each generation must own its own DADDyState.
"""
import torch
from .sampler import Sampler
from .custom import CustomSampler, SS_Fused
from .presets import DefaultSampler
from ...ext import exllamav3_ext as ext


class DADDyState:
    def __init__(self):
        self.balances = None
        self.counts = None
        self.output = None
        self.workspace = None
        self.before = None

    def prepare(self, device, bsz):
        if self.balances is not None:
            if self.balances.device != device or self.balances.shape[0] != bsz:
                raise ValueError("Reset DADDy state before changing device or batch size")
            return
        if not hasattr(ext, "daddy_sampler") or getattr(ext, "DADDY_SAMPLER_COUNTS", None) != 3:
            raise RuntimeError("DADDy requires rebuilding the native exllamav3 extension")
        self.balances = torch.zeros((bsz, 8), dtype = torch.double, device = device)
        self.counts = torch.zeros((bsz, ext.DADDY_SAMPLER_COUNTS), dtype = torch.long, device = device)
        self.output = torch.empty((bsz, 1), dtype = torch.long, device = device)
        self.workspace = torch.empty(
            bsz * ext.FUSED_SAMPLER_MAX_BLOCKS * ext.DADDY_SAMPLER_PARTIAL_FLOATS,
            dtype = torch.float, device = device,
        )

    def launch(self, logits, state, fused, workspace, histogram):
        self.prepare(logits.device, state.bsz)
        bits = state.fused_mask is not None and state.fused_mask.dtype == torch.int32
        ext.daddy_sampler(
            logits, None if bits else state.fused_mask, state.fused_mask if bits else None,
            self.output, workspace, state.fused_dim or state.dim,
            fused.inv_temp, fused.minp_log, state.rand_u32, fused.mode, fused.filters,
            fused.top_k, fused.top_p, fused.inv_temp_filter, histogram,
            self.workspace, self.balances, self.counts, state.speculative_token, 4,
        )
        return self.output

    def force(self, token):
        # A singleton eligible distribution adds and spends one unit in bin 7.
        # Its net balance update is zero, including when that bin is overfilled.
        self.output.copy_(token.reshape_as(self.output))
        return self.output

    def save_before_step(self):
        """Reusable device snapshot, needed only by jobs using banned strings."""
        if self.before is None:
            self.before = (torch.empty_like(self.balances), torch.empty_like(self.counts))
        self.before[0].copy_(self.balances)
        self.before[1].copy_(self.counts)

    def checkpoint(self):
        return tuple(t.clone() for t in self.before)

    def restore(self, snapshot):
        self.balances.copy_(snapshot[0])
        self.counts.copy_(snapshot[1])

    def stats(self):
        """Explicit diagnostic readback, outside the sampling hot path."""
        if self.balances is None:
            return []
        balances, counts = self.balances.cpu().tolist(), self.counts.cpu().tolist()
        return [dict(
            balances=b, direct_proposals=c[0], proposal_draws=c[1], all_full_fallbacks=c[2],
        ) for b, c in zip(balances, counts)]


class DADDySampler(Sampler):
    """Stateful probability-bin selection over a fused-compatible sampler.

    Jobs automatically create independent generation states. Direct forward()
    calls use one persistent stream per batch row until reset() is called. The
    returned tensor is a reusable buffer, overwritten by the next call: clone
    it if it needs to outlive that call. No exact distribution preservation is
    claimed by this heuristic policy.
    """
    def __init__(self, sampler=None):
        super().__init__()
        self.sampler = sampler if sampler is not None else DefaultSampler()
        if (not isinstance(self.sampler, CustomSampler) or not self.sampler.steps or
            not isinstance(self.sampler.steps[-1], SS_Fused)):
            raise ValueError("DADDySampler requires a fused-compatible sampler pipeline")
        self.reqs_past_ids = self.sampler.reqs_past_ids
        self.reqs_torch_seed = self.sampler.reqs_torch_seed
        self.reset()

    def new_generation(self):
        return DADDyState()

    def reset(self):
        self.state = self.new_generation()

    @torch.inference_mode()
    def forward(self, logits, sequence_ids=None, rand_u32=None, tokenizer=None,
                logit_mask=None, return_state=False, *, generation_state=None,
                speculative_token=None, forced_token=None):
        state = self.state if generation_state is None else generation_state
        if forced_token is not None:
            if return_state:
                raise ValueError("return_state is not supported with forced_token")
            state.prepare(logits.device, logits.numel() // logits.shape[-1])
            return state.force(forced_token).view(logits.shape[:-1])
        return self.sampler.forward(
            logits, sequence_ids, rand_u32, tokenizer, logit_mask, return_state,
            daddy_state=state, speculative_token=speculative_token,
        )
