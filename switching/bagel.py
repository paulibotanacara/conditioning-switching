"""Conditioning switching using the official BAGEL model and public checkpoint.

Context preparation and flow calls follow ByteDance-Seed/Bagel (Apache-2.0).
Copyright 2025 Bytedance Ltd. and/or its affiliates.
"""

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib
import math
import sys

import numpy as np
from PIL import Image
import torch

from .schedule import EDIT, Schedule
from .policy import decide

MODEL_ID = 'ByteDance-Seed/BAGEL-7B-MoT'
SOURCE_REVISION = 'a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f'
SELECTIVE_THRESHOLDS = (0.166, 0.469)


def load_pipeline(source_dir, model=MODEL_ID, *, device='cuda:0', cache_dir=None,
                  revision=None, local_files_only=False):
    """Use an official BAGEL checkout; download weights unless model is a directory."""
    source_dir = Path(source_dir).resolve()
    if not (source_dir / 'modeling/bagel/bagel.py').is_file():
        raise ValueError('source_dir must point to the official ByteDance-Seed/Bagel checkout')
    sys.path.insert(0, str(source_dir))
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from huggingface_hub import snapshot_download
    from data.data_utils import add_special_tokens
    from data.transforms import ImageTransform
    from modeling.autoencoder import load_ae
    from modeling.bagel import (Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM,
                                SiglipVisionConfig, SiglipVisionModel)
    from modeling.bagel.qwen2_navit import NaiveCache
    from modeling.qwen2 import Qwen2Tokenizer
    directory = Path(model)
    if not directory.is_dir():
        directory = Path(snapshot_download(model, cache_dir=cache_dir, revision=revision,
                         local_files_only=local_files_only,
                         allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model']))
    llm = Qwen2Config.from_json_file(str(directory / 'llm_config.json'))
    llm.qk_norm, llm.tie_word_embeddings, llm.layer_module = True, False, 'Qwen2MoTDecoderLayer'
    vit = SiglipVisionConfig.from_json_file(str(directory / 'vit_config.json'))
    vit.rope = False
    vit.num_hidden_layers -= 1
    vae, vae_config = load_ae(local_path=str(directory / 'ae.safetensors'))
    config = BagelConfig(visual_gen=True, visual_und=True, llm_config=llm,
                        vit_config=vit, vae_config=vae_config, vit_max_num_patch_per_side=70,
                        connector_act='gelu_pytorch_tanh', latent_patch_size=2, max_latent_size=64)
    with init_empty_weights():
        model = Bagel(Qwen2ForCausalLM(llm), SiglipVisionModel(vit), config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit, meta=True)
    model = load_checkpoint_and_dispatch(model, str(directory / 'ema.safetensors'),
                                         device_map={'': device}, dtype=torch.bfloat16).eval()
    model.language_model.model.enable_taylorseer = False
    tokenizer = Qwen2Tokenizer.from_pretrained(str(directory))
    tokenizer, token_ids, _ = add_special_tokens(tokenizer)
    return SimpleNamespace(model=model, vae=vae.to(device=device, dtype=torch.float32).eval(),
                           tokenizer=tokenizer, token_ids=token_ids, cache_class=NaiveCache,
                           vae_transform=ImageTransform(1024, 448, 16),
                           vit_transform=ImageTransform(980, 224, 14), device=torch.device(device))


def _move(inputs, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()}


def _empty(pipe):
    return dict(kv_lens=[0], ropes=[0], past_key_values=pipe.cache_class(
        pipe.model.config.llm_config.num_hidden_layers))


def _text(pipe, text, context):
    inputs, lengths, ropes = pipe.model.prepare_prompts(
        curr_kvlens=context['kv_lens'], curr_rope=context['ropes'], prompts=[text],
        tokenizer=pipe.tokenizer, new_token_ids=pipe.token_ids)
    cache = pipe.model.forward_cache_update_text(context['past_key_values'], **_move(inputs, pipe.device))
    return dict(kv_lens=lengths, ropes=ropes, past_key_values=cache)


def _image(pipe, image):
    context = _empty(pipe)
    for kind, transform in [('vae', pipe.vae_transform), ('vit', pipe.vit_transform)]:
        inputs, lengths, ropes = getattr(pipe.model, f'prepare_{kind}_images')(
            curr_kvlens=context['kv_lens'], curr_rope=context['ropes'], images=[image],
            transforms=transform, new_token_ids=pipe.token_ids)
        args = (pipe.vae, context['past_key_values']) if kind == 'vae' else (context['past_key_values'],)
        cache = getattr(pipe.model, f'forward_cache_update_{kind}')(*args, **_move(inputs, pipe.device))
        context = dict(kv_lens=lengths, ropes=ropes, past_key_values=cache)
    return context


def _inputs(pipe, context, negative_text, negative_image, shape):
    main = pipe.model.prepare_vae_latent(curr_kvlens=context['kv_lens'], curr_rope=context['ropes'],
        image_sizes=[shape], new_token_ids=pipe.token_ids)
    noise = main.pop('packed_init_noises').to(pipe.device)
    inputs = _move(main, pipe.device)
    inputs['past_key_values'] = context['past_key_values']
    for branch, ctx in [('text', negative_text), ('img', negative_image)]:
        cfg = pipe.model.prepare_vae_latent_cfg(curr_kvlens=ctx['kv_lens'], curr_rope=ctx['ropes'],
                                              image_sizes=[shape])
        inputs.update({key.replace('cfg_', f'cfg_{branch}_', 1): value
                       for key, value in _move(cfg, pipe.device).items()})
        inputs[f'cfg_{branch}_past_key_values'] = ctx['past_key_values']
    return inputs, noise


def source_attention(query, key, query_indexes, n_source, chunk_size=128):
    """Source-prefix attention mass for packed latent queries, including GQA."""
    query = query[query_indexes].bfloat16()
    key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1).bfloat16()
    chunks = []
    for start in range(0, len(query), chunk_size):
        logits = torch.einsum('qhd,khd->hqk', query[start:start + chunk_size], key)
        mass = (logits * (1 / math.sqrt(query.shape[-1]))).softmax(-1)[..., :n_source]
        chunks.append(mass.sum(-1).mean(0).float().cpu().numpy())
    return np.concatenate(chunks)


@contextmanager
def capture_attention(model, query_indexes, n_source, maps):
    """Capture the conditional pass only; restore FlashAttention even on failure."""
    module = importlib.import_module(type(model.language_model).__module__)
    original = module.flash_attn_varlen_func
    n_layers = len(model.language_model.model.layers)
    def attend(*args, **kwargs):
        if len(maps) < n_layers:
            q = kwargs['q'] if 'q' in kwargs else args[0]
            k = kwargs['k'] if 'k' in kwargs else args[1]
            maps.append(source_attention(q, k, query_indexes, n_source))
        return original(*args, **kwargs)
    module.flash_attn_varlen_func = attend
    try:
        yield
    finally:
        module.flash_attn_varlen_func = original


def denoise(pipe, prepared, latents, schedule, *, guidance_scale, image_guidance_scale,
            timestep_shift, cfg_renorm_min, grid_shape, n_source, diagnostics):
    model = pipe.model
    active = schedule
    selective = schedule.mode.endswith('_selective')
    maps = []
    # N updates require N+1 endpoints (the paper branch uses this convention).
    times = torch.linspace(1, 0, schedule.steps + 1, device=latents.device)
    times = timestep_shift * times / (1 + (timestep_shift - 1) * times)
    for step, (t, next_t) in enumerate(zip(times[:-1], times[1:])):
        if selective and step == 10:
            if len(maps) != len(model.language_model.model.layers):
                raise RuntimeError('Selective probe did not capture every BAGEL layer')
            active, info = decide(np.stack(maps).mean(0).reshape(grid_shape), schedule, SELECTIVE_THRESHOLDS)
            if diagnostics is not None:
                diagnostics.update(info)
        condition = EDIT if selective and step <= 9 else active.at(step)
        inputs = prepared[condition]
        kwargs = dict(x_t=latents, timestep=t.expand(len(latents)), **inputs,
                      cfg_text_scale=guidance_scale, cfg_img_scale=image_guidance_scale,
                      cfg_renorm_min=cfg_renorm_min,
                      cfg_renorm_type='text_channel' if condition.source else 'global')
        if selective and step == 9:
            with capture_attention(model, inputs['packed_vae_token_indexes'], n_source, maps):
                velocity = model._forward_flow(**kwargs)
        else:
            velocity = model._forward_flow(**kwargs)
        latents = latents - velocity.to(latents.device) * (t - next_t)
    return latents


@torch.inference_mode()
def generate(pipe, image, instruction, *, caption=None, improved_instruction=None,
             schedule=None, seed=1337, guidance_scale=4.0, image_guidance_scale=1.5,
             timestep_shift=3.0, cfg_renorm_min=1.0, diagnostics=None):
    schedule = schedule or Schedule()
    selective = schedule.mode.endswith('_selective')
    if selective and schedule.steps <= 10:
        raise ValueError('Selective switching needs more than 10 steps')
    if diagnostics is not None:
        diagnostics.clear()
    if (not all(math.isfinite(v) for v in (guidance_scale, image_guidance_scale, timestep_shift, cfg_renorm_min))
            or guidance_scale <= 1 or image_guidance_scale < 1 or timestep_shift <= 0
            or not 0 <= cfg_renorm_min <= 1):
        raise ValueError('Require finite text guidance > 1, image guidance >= 1, positive shift, and renorm in [0,1]')
    prompts = dict(instruction=instruction, caption=caption, improved_instruction=improved_instruction)
    conditions = {schedule.at(k) for k in range(schedule.steps)}
    if selective:
        conditions.add(EDIT)
    for condition in conditions:
        if not isinstance(prompts[condition.text], str) or not prompts[condition.text].strip():
            raise ValueError(f'{schedule.mode} requires non-empty {condition.text}')
    image = pipe.vae_transform.resize_transform(image.convert('RGB'))
    shape = image.size[::-1]
    device_index = pipe.device.index if pipe.device.index is not None else torch.cuda.current_device()
    with torch.random.fork_rng(devices=[device_index]), torch.autocast('cuda', dtype=torch.bfloat16):
        torch.random.default_generator.manual_seed(seed)
        with torch.cuda.device(pipe.device):
            torch.cuda.manual_seed(seed)
        source = _image(pipe, image) if any(c.source for c in conditions) else _empty(pipe)
        prepared = {}
        latents = None
        for condition in sorted(conditions, key=lambda c: (c.text, c.source)):
            negative = source if condition.source else _empty(pipe)
            positive = _text(pipe, prompts[condition.text], deepcopy(negative))
            image_negative = _text(pipe, prompts[condition.text], _empty(pipe)) if condition.source else positive
            inputs, noise = _inputs(pipe, positive, negative, image_negative, shape)
            prepared[condition] = inputs
            if latents is None:
                latents = noise
        grid_shape = tuple(size // pipe.model.latent_downsample for size in shape)
        latents = denoise(pipe, prepared, latents, schedule, guidance_scale=guidance_scale,
                          image_guidance_scale=image_guidance_scale, timestep_shift=timestep_shift,
                          cfg_renorm_min=cfg_renorm_min, grid_shape=grid_shape,
                          n_source=source['kv_lens'][0], diagnostics=diagnostics)
        h, w = grid_shape
        patch, channels = pipe.model.latent_patch_size, pipe.model.latent_channel
        latent = latents.reshape(1, h, w, patch, patch, channels)
        latent = torch.einsum('nhwpqc->nchpwq', latent).reshape(1, channels, h * patch, w * patch)
    pixels = pipe.vae.decode(latent.float()).mul(0.5).add(0.5).clamp(0, 1)[0]
    return Image.fromarray((pixels.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy())
