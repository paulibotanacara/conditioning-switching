# Sampling/preparation adapted from Hugging Face Diffusers' pipeline_flux2_klein.py.
# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (see THIRD_PARTY_NOTICES.md).
"""A single-image reference sampler on top of the public FLUX.2 Klein pipeline.

Defaults to Diffusers' native preprocessing, empty-text CFG and Euler schedule.
The optional paper sampling profile matches the analysis-script conventions.
This demonstrates conditioning switching, not exact reproduction of paper scores.
No cross-step attention cache is enabled. Do not enable one on the supplied model.
"""

import math

import numpy as np
import torch
from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
from PIL import Image

from .schedule import Schedule, EDIT

MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"


def load_pipeline(model=MODEL_ID, *, device="cuda", cpu_offload=False, **kwargs):
    """Load the undistilled checkpoint. kwargs accepts revision/local_files_only."""
    pipe = Flux2KleinPipeline.from_pretrained(model, torch_dtype=torch.bfloat16, **kwargs)
    if pipe.config.is_distilled:
        raise ValueError("Use a klein-base checkpoint, not the four-step distilled model")
    if cpu_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    return pipe


def denoise(pipe, latents, latent_ids, image_latents, image_ids, texts, negative,
            schedule, timesteps, guidance_scale, reference_times=None, probe=None, guidance=None):
    """Keep one evolving latent; only text and source-token availability change."""
    if len(timesteps) != schedule.steps:
        raise ValueError("The scheduler must produce exactly one timestep per schedule step")
    active_schedule = schedule
    pipe.scheduler.set_begin_index(0)
    with pipe.progress_bar(total=schedule.steps) as progress:
        for k, t in enumerate(timesteps):
            if probe is not None and k == 10:
                active_schedule = probe.decide(schedule)
            conditioning = EDIT if probe is not None and k <= 9 else active_schedule.at(k)
            if guidance is not None and guidance.start <= k < guidance.end:
                endpoint = guidance.endpoint(guidance_scale)
                if endpoint == 't2i':
                    from .schedule import T2I
                    conditioning = T2I
                elif endpoint != 'editing':
                    from .guidance import predict as guided_prediction
                    velocity = guided_prediction(pipe, latents, latent_ids, image_latents,
                        image_ids, texts, negative, t, guidance_scale, guidance,
                        paper=reference_times is not None)
                    if reference_times is not None:
                        latents = latents + (reference_times[k + 1] - t).reshape(1, 1, 1) * velocity
                    else:
                        latents = pipe.scheduler.step(velocity, t, latents, return_dict=False)[0]
                    progress.update()
                    continue
            positive_forward = probe.forward if probe is not None and k == 9 else None
            def predict(**kwargs):
                if positive_forward is not None:
                    return positive_forward(pipe.transformer, **kwargs)
                return pipe.transformer(**kwargs)
            embeddings, text_ids = texts[conditioning.text]
            # Source tokens must be present/absent in BOTH CFG branches.
            model_input, model_ids = latents, latent_ids
            if conditioning.source:
                model_input = torch.cat((latents, image_latents), dim=1)
                model_ids = torch.cat((latent_ids, image_ids), dim=1)
            common = dict(
                hidden_states=model_input.to(pipe.transformer.dtype),
                timestep=t.expand(latents.shape[0]).to(latents.dtype) / 1000,
                guidance=None,
                img_ids=model_ids,
                return_dict=False,
            )
            if reference_times is not None:
                common["timestep"] = t.expand(latents.shape[0]).to(pipe.transformer.dtype)
                common["hidden_states"] = common["hidden_states"].repeat(2, 1, 1)
                common["img_ids"] = common["img_ids"].repeat(2, 1, 1)
                common["timestep"] = common["timestep"].repeat(2)
                # The original implementation evaluates CFG as one batch of two.
                predictions = predict(
                    **common, encoder_hidden_states=torch.cat((embeddings, negative[0])),
                    txt_ids=torch.cat((text_ids, negative[1])),
                )[0][:, :latents.shape[1]]
                positive, uncond = predictions.chunk(2)
                velocity = uncond + guidance_scale * (positive - uncond)
                dt = (reference_times[k + 1] - t).reshape(1, 1, 1)
                # Float32 Euler state, while denoiser/CFG predictions are bfloat16.
                latents = latents + dt * velocity
                progress.update()
                continue
            # Each step evaluates only the active conditioning mode.
            positive = predict(
                **common, encoder_hidden_states=embeddings, txt_ids=text_ids,
            )[0][:, :latents.shape[1]]
            uncond = pipe.transformer(
                **common, encoder_hidden_states=negative[0], txt_ids=negative[1],
            )[0][:, :latents.shape[1]]
            velocity = uncond + guidance_scale * (positive - uncond)
            latents = pipe.scheduler.step(velocity, t, latents, return_dict=False)[0]
            progress.update()
    return latents


@torch.inference_mode()
def generate(pipe, image: Image.Image, instruction: str, *, caption=None,
             improved_instruction=None, schedule=None, seed=0, guidance_scale=4.0,
             height=None, width=None, max_sequence_length=512, sampling="diffusers",
             negative_prompt_embeds=None, diagnostics=None,
             generator_device="cpu", selective_thresholds=(0.50, 0.22), guidance=None):
    """Generate one PIL image, using a supplied caption (no VLM/API call).

    Resets the RNG for each call, so repeated calls with the same source/seed use
    the same initial noise. Output size defaults to the preprocessed source size.
    A zero-length interval is an identity schedule, useful for baseline checks.
    With pure_editing, guidance=Guidance(...) applies bounded continuous guidance.
    sampling="paper" hashes the base seed with the instruction, draws noise on
    the execution device before patchification, and uses the original Euler grid.
    Supply negative_prompt_embeds to reuse the experiment's fixed null embedding.
    For selective mode, diagnostics receives the step-9 maps, CV, pi, decision,
    and actual interval (which can start no earlier than step 10).
    """
    schedule = schedule or Schedule()
    selective = schedule.mode == "editing_t2i_editing_selective"
    if selective and schedule.steps <= 10:
        raise ValueError("Selective switching needs more than 10 steps (probe at step 9)")
    if diagnostics is not None:
        diagnostics.clear()
    if sampling not in ("diffusers", "paper"):
        raise ValueError("sampling must be 'diffusers' or 'paper'")
    if pipe.config.is_distilled:
        raise ValueError("This sampler requires an undistilled klein-base checkpoint")
    if not math.isfinite(guidance_scale) or guidance_scale <= 1:
        raise ValueError("guidance_scale must be finite and > 1 (two-pass CFG)")
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image")
    required_texts = schedule.required_texts()
    if guidance is not None:
        guidance.validate(schedule.steps)
        if schedule.mode != 'pure_editing':
            raise ValueError('Use a pure_editing schedule with Guidance; its interval controls guidance')
        if guidance.needs_caption(guidance_scale):
            required_texts.add('caption')
    prompts = dict(instruction=instruction, caption=caption,
                   improved_instruction=improved_instruction)
    for name in required_texts:
        if not isinstance(prompts[name], str) or not prompts[name].strip():
            raise ValueError(f"{schedule.mode} requires non-empty {name}")

    image = image.convert("RGB")
    pipe.image_processor.check_image_input(image)
    if image.width * image.height > 1024 * 1024:
        image = pipe.image_processor._resize_to_target_area(image, 1024 * 1024)
    multiple = pipe.vae_scale_factor * 2
    source_width = image.width // multiple * multiple
    source_height = image.height // multiple * multiple
    height = source_height if height is None else height
    width = source_width if width is None else width
    for dimension in (source_width, source_height, width, height):
        if type(dimension) is not int or dimension < multiple or dimension % multiple:
            raise ValueError(f"Image dimensions must be positive multiples of {multiple}")

    device = pipe._execution_device
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    try:
        texts = {
            key: pipe.encode_prompt(prompt=prompts[key], device=device,
                                    max_sequence_length=max_sequence_length)
            for key in sorted(required_texts)
        }
        negative = pipe.encode_prompt(prompt="", device=device,
                                      prompt_embeds=negative_prompt_embeds,
                                      max_sequence_length=max_sequence_length)
        negative = (negative[0].to(device=device, dtype=pipe.transformer.dtype),
                    negative[1].to(device))
        source = pipe.image_processor.preprocess(
            image, height=source_height, width=source_width, resize_mode="crop")
        latents, latent_ids = pipe.prepare_latents(
            batch_size=1, num_latents_channels=pipe.transformer.config.in_channels // 4,
            height=height, width=width, dtype=negative[0].dtype,
            device=device, generator=generator, latents=None,
        )
        image_latents, image_ids = pipe.prepare_image_latents(
            images=[source], batch_size=1, generator=generator,
            device=device, dtype=pipe.vae.dtype,
        )
        reference_times = None
        if sampling == "paper":
            from .reference import paper_noise, paper_times
            if any(text[0].shape != negative[0].shape for text in texts.values()):
                raise ValueError("Paper CFG requires equally padded positive and negative embeddings")
            latents, latent_ids = paper_noise(
                pipe, seed, instruction, height, width, device, pipe.transformer.dtype)
            reference_times = paper_times(schedule.steps, latents.shape[1], device)
        sigmas = np.linspace(1.0, 1 / schedule.steps, schedule.steps)
        if getattr(pipe.scheduler.config, "use_flow_sigmas", False):
            sigmas = None
        if reference_times is None:
            timesteps, _ = retrieve_timesteps(
                pipe.scheduler, schedule.steps, device, sigmas=sigmas,
                mu=compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=schedule.steps),
            )
        else:
            timesteps = reference_times[:-1]
        probe = None
        if selective:
            from .selective import AttentionProbe
            probe = AttentionProbe(latents.shape[1], image_latents.shape[1],
                                   (height // multiple, width // multiple), selective_thresholds)
        latents = denoise(pipe, latents, latent_ids, image_latents, image_ids,
                          texts, negative, schedule, timesteps, guidance_scale, reference_times, probe, guidance)
        if diagnostics is not None and probe is not None:
            diagnostics.update(probe.result)
        if sampling == "paper":
            latents = latents.to(pipe.vae.dtype)
        latents = pipe._unpack_latents_with_ids(latents, latent_ids)
        mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents)
        std = (pipe.vae.bn.running_var.view(1, -1, 1, 1)
               + pipe.vae.config.batch_norm_eps).sqrt().to(latents)
        latents = pipe._unpatchify_latents(latents * std + mean)
        decoded = pipe.vae.decode(latents, return_dict=False)[0]
        if sampling == "paper":
            pixels = decoded[0].mul(0.5).add(0.5).clamp(0, 1).float()
            return Image.fromarray((pixels.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        return pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    finally:
        pipe.maybe_free_model_hooks()
