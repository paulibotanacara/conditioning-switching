# Attention processors adapted from Hugging Face Diffusers (Apache-2.0).
# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
"""Source attention from the conditional forward at step 9."""

import math
import numpy as np
import torch
from diffusers.models.transformers.transformer_flux2 import (
    _get_qkv_projections, apply_rotary_emb, dispatch_attention_fn,
)

class Flux2AttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self, probe, original):
        self.probe = probe
        self._attention_backend = original._attention_backend
        self._parallel_config = original._parallel_config

    def __call__(
        self,
        attn: "Flux2Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = self.probe.attend(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class Flux2ParallelSelfAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self, probe, original):
        self.probe = probe
        self._attention_backend = original._attention_backend
        self._parallel_config = original._parallel_config

    def __call__(
        self,
        attn: "Flux2ParallelSelfAttention",
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Parallel in (QKV + MLP in) projection
        hidden_states = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            hidden_states, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )

        # Handle the attention logic
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = self.probe.attend(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # Handle the feedforward (FF) logic
        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)

        # Concatenate and parallel output projection
        hidden_states = torch.cat([hidden_states, mlp_hidden_states], dim=-1)
        hidden_states = attn.to_out(hidden_states)

        return hidden_states


def source_attention(query, key, n_text, n_output, n_source, chunk_size=128):
    """Head-averaged source mass; only conditional batch element zero is used."""
    key = key[:1].to(torch.bfloat16)
    maps = []
    for start in range(0, n_output, chunk_size):
        q = query[:1, n_text + start:n_text + min(start + chunk_size, n_output)]
        logits = torch.einsum('bqhd,bkhd->bhqk', q.to(torch.bfloat16), key)
        logits = logits * (1 / math.sqrt(query.shape[-1]))
        mass = logits.softmax(-1)[..., n_text + n_output:n_text + n_output + n_source]
        maps.append(mass.sum(-1).mean(1)[0].float().cpu().numpy())
    return np.concatenate(maps)


class AttentionProbe:
    """Temporarily replace attention processors for one ordinary denoiser call."""

    def __init__(self, n_output, n_source, grid_shape, thresholds=(0.50, 0.22)):
        self.n_output = n_output
        self.n_source = n_source
        self.grid_shape = grid_shape
        self.thresholds = thresholds
        self.maps = []
        self.result = None

    def attend(self, query, key, value, **kwargs):
        if kwargs.get('attn_mask') is not None:
            raise ValueError('Selective probing requires unmasked Klein attention')
        self.maps.append(source_attention(query, key, self.n_text, self.n_output, self.n_source))
        return dispatch_attention_fn(query, key, value, **kwargs)

    def forward(self, transformer, **kwargs):
        self.n_text = kwargs['encoder_hidden_states'].shape[1]
        originals = dict(transformer.attn_processors)
        replacements = {}
        for name, original in originals.items():
            cls = (Flux2ParallelSelfAttnProcessor if name.startswith('single_transformer_blocks.')
                   else Flux2AttnProcessor)
            replacements[name] = cls(self, original)
        try:
            transformer.set_attn_processor(replacements)
            return transformer(**kwargs)
        finally:
            transformer.set_attn_processor(originals)

    def decide(self, schedule):
        if not self.maps:
            raise RuntimeError('No source attention was captured')
        attention = np.stack(self.maps).mean(0).astype(np.float64).reshape(self.grid_shape)
        from .policy import decide
        chosen, self.result = decide(attention, schedule, self.thresholds)
        return chosen
