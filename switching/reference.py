"""Sampling conventions used by the paper's 4B analysis scripts.

Kept separate from the default Diffusers example. Matching these conventions is
necessary, but not sufficient, for pixel-identical outputs across implementations.
"""

import hashlib
import math

import torch
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu


def prompt_seed(seed, instruction):
    return abs(seed + int(hashlib.sha1(instruction.encode("utf-8")).hexdigest(), 16)) % 2**32


def paper_times(steps, image_tokens, device):
    # The analysis scripts specify shift=3, but dynamic shifting overrides it.
    # Their uniform time distribution stops at numerical-safety endpoints.
    times = torch.linspace(0.999, 1e-5, steps + 1, device=device, dtype=torch.float32)
    mu = compute_empirical_mu(image_tokens, steps)
    return math.exp(mu) / (math.exp(mu) + (1 / times - 1))


def paper_noise(pipe, seed, instruction, height, width, device, dtype):
    # The internal editor uses randn_like on the unpatchified source latent,
    # on CUDA, rather than CPU noise in the already-patchified layout.
    rng = torch.Generator(device=device).manual_seed(prompt_seed(seed, instruction))
    raw = torch.randn(
        (1, pipe.transformer.config.in_channels // 4,
         height // pipe.vae_scale_factor, width // pipe.vae_scale_factor),
        generator=rng, device=device, dtype=dtype,
    )
    patches = pipe._patchify_latents(raw)
    return pipe._pack_latents(patches), pipe._prepare_latent_ids(patches).to(device)
