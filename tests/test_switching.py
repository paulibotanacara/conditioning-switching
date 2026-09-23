"""CPU integration tests with tiny real Diffusers models; no downloads required."""

import types

import numpy as np
import pytest
import torch
from diffusers import (AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler,
                       Flux2KleinPipeline, Flux2Transformer2DModel)
from PIL import Image

from switching.flux import generate, load_pipeline
from switching.schedule import MODES, Schedule


@pytest.fixture
def pipe():
    torch.set_num_threads(1)
    torch.manual_seed(123)
    transformer = Flux2Transformer2DModel(
        in_channels=16, num_layers=1, num_single_layers=1,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=24,
        axes_dims_rope=(4, 4, 4, 4), guidance_embeds=False,
    )
    vae = AutoencoderKLFlux2(
        block_out_channels=(32,), down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",), latent_channels=4,
        layers_per_block=1, norm_num_groups=8, sample_size=8,
    ).eval()
    result = Flux2KleinPipeline(
        transformer=transformer.eval(), vae=vae,
        scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
        text_encoder=None, tokenizer=None, is_distilled=False,
    )

    # Deterministic stand-in for the expensive text encoder. Different lengths
    # exercise switching text position IDs as well as embeddings.
    def encode(self, prompt, device=None, **kwargs):
        length = 2 + len(prompt) % 3
        g = torch.Generator().manual_seed(sum(map(ord, prompt)))
        embeddings = torch.randn(1, length, 24, generator=g).to(device)
        return embeddings, self._prepare_text_ids(embeddings).to(device)

    result.encode_prompt = types.MethodType(encode, result)
    # Tiny fixture only: upstream's real-image minimum is 64 pixels.
    check_image = result.image_processor.check_image_input
    result.image_processor.check_image_input = lambda image: check_image(image, min_side_length=2)
    result.set_progress_bar_config(disable=True)
    return result


@pytest.fixture
def source():
    return Image.new("RGB", (8, 8), (120, 70, 30))


def test_matches_upstream_pure_editing(pipe, source):
    """Check preprocessing, RNG order, CFG, schedule, latent packing and decode."""
    expected = pipe(
        image=source, prompt="edit", num_inference_steps=4, guidance_scale=4.0,
        generator=torch.Generator(device="cpu").manual_seed(7),
    ).images[0]
    actual = generate(pipe, source, "edit", schedule=Schedule("pure_editing", 4), seed=7)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("mode", [m for m in MODES if not m.endswith("selective")])
def test_modes_execute_and_restore_conditioning(pipe, source, mode):
    calls = []

    def record(module, args, kwargs):
        calls.append((kwargs["hidden_states"].shape[1],
                      kwargs["encoder_hidden_states"].detach().clone(),
                      kwargs["img_ids"].shape[1], kwargs["txt_ids"].shape[1]))

    handle = pipe.transformer.register_forward_pre_hook(record, with_kwargs=True)
    schedule = Schedule(mode, 4, 1, 3)
    prompts = {"instruction": "edit", "caption": "scene", "improved_instruction": "improved"}
    try:
        image = generate(pipe, source, "edit", caption="scene", improved_instruction="improved",
                         schedule=schedule)
    finally:
        handle.remove()
    assert image.size == (8, 8)
    assert len(calls) == 8  # Exactly two evaluations per step, for every mode.
    for k in range(4):
        cond = schedule.at(k)
        positive, negative = calls[k * 2:k * 2 + 2]
        # Tiny VAE: 8x8 -> sixteen 2x2 latent patches; source adds sixteen.
        expected_tokens = 32 if cond.source else 16
        assert positive[0] == negative[0] == expected_tokens
        assert positive[2] == negative[2] == expected_tokens
        expected = pipe.encode_prompt(prompts[cond.text], device="cpu")[0]
        torch.testing.assert_close(positive[1], expected)
        torch.testing.assert_close(negative[1], pipe.encode_prompt("", device="cpu")[0])
        assert positive[3] == positive[1].shape[1]
        assert negative[3] == negative[1].shape[1]


def test_empty_interval_equals_baseline_and_calls_are_independent(pipe, source):
    baseline = generate(pipe, source, "edit", schedule=Schedule("pure_editing", 4))
    generate(pipe, source, "edit", caption="scene", schedule=Schedule(steps=4, start=0, end=4))
    identity = generate(pipe, source, "edit", schedule=Schedule(steps=4, start=2, end=2))
    np.testing.assert_array_equal(np.asarray(baseline), np.asarray(identity))


def test_boundaries():
    s = Schedule()
    assert s.at(9).source and not s.at(10).source and not s.at(15).source and s.at(16).source
    assert sum(not s.at(k).source for k in range(50)) == 6
    assert Schedule(start=0, end=50).required_texts() == {"caption"}
    assert Schedule(start=50, end=50).required_texts() == {"instruction"}
    assert Schedule(start=0, end=50).required_texts() == {"caption"}
    with pytest.raises(ValueError):
        Schedule(start=16, end=10)
    with pytest.raises(ValueError):
        Schedule(steps=0)


def test_missing_caption_fails_before_inference(pipe, source):
    with pytest.raises(ValueError, match="caption"):
        generate(pipe, source, "edit")


def test_distilled_model_rejected(pipe, source):
    pipe.register_to_config(is_distilled=True)
    with pytest.raises(ValueError, match="undistilled"):
        generate(pipe, source, "edit", caption="scene")


def test_load_pipeline_from_local_checkpoint(pipe, tmp_path):
    pipe.save_pretrained(tmp_path)
    loaded = load_pipeline(str(tmp_path), device="cpu", local_files_only=True,
                           text_encoder=None, tokenizer=None)
    assert isinstance(loaded, Flux2KleinPipeline)
    assert not loaded.config.is_distilled
    assert loaded.transformer.dtype == torch.bfloat16
    assert loaded.transformer.device.type == "cpu"


@pytest.mark.parametrize("mode", [m for m in MODES if not m.endswith("selective")])
def test_paper_profile_modes(pipe, source, mode):
    encode = pipe.encode_prompt

    def padded(prompt, device=None, prompt_embeds=None, **kwargs):
        embeddings = encode(prompt, device=device)[0] if prompt_embeds is None else prompt_embeds
        embeddings = torch.nn.functional.pad(embeddings, (0, 0, 0, 4 - embeddings.shape[1]))
        return embeddings, pipe._prepare_text_ids(embeddings).to(device)

    pipe.encode_prompt = padded
    calls = []

    def record(module, args, kwargs):
        calls.append((kwargs["hidden_states"].shape, kwargs["timestep"].clone()))

    hook = pipe.transformer.register_forward_pre_hook(record, with_kwargs=True)
    schedule = Schedule(mode, 4, 1, 3)
    try:
        result = generate(pipe, source, "edit", caption="scene", improved_instruction="improved",
                          schedule=schedule, seed=1337, sampling="paper",
                          negative_prompt_embeds=torch.zeros(1, 4, 24))
    finally:
        hook.remove()
    assert result.size == source.size
    assert len(calls) == 4  # One batch containing both CFG predictions per step.
    for k, (shape, timestep) in enumerate(calls):
        assert shape[0] == 2
        assert shape[1] == (32 if schedule.at(k).source else 16)
        assert 0 < timestep[0] < 1


def test_paper_seed_and_grid():
    from switching.reference import prompt_seed, paper_times
    assert prompt_seed(1337, "Replace the blue bird in the image with a red fox.") == 1796810183
    times = paper_times(50, 4096, "cpu")
    assert times.dtype == torch.float32 and len(times) == 51
    assert torch.all(times[:-1] > times[1:])
    assert times[0] < 1 and times[-1] > 0


@pytest.mark.parametrize('mode', MODES)
def test_full_interval(mode):
    schedule = Schedule(mode, 50, 0, 50)
    assert schedule.at(0) == schedule.at(49) == MODES[mode][1]


@pytest.mark.parametrize('accept', [False, True])
@pytest.mark.parametrize('start,end', [(10, 12), (0, 12), (0, 5)])
def test_selective_decision_matches_fixed_run(pipe, source, monkeypatch, accept, start, end):
    from switching.selective import AttentionProbe
    original = AttentionProbe.attend
    def controlled(self, *args, **kwargs):
        output = original(self, *args, **kwargs)
        self.maps[-1][:] = 0.1 if accept else 0.3
        return output
    monkeypatch.setattr(AttentionProbe, 'attend', controlled)
    processors = dict(pipe.transformer.attn_processors)
    calls = []
    handle = pipe.transformer.register_forward_pre_hook(lambda *args: calls.append(1))
    diagnostics = {}
    actual = generate(pipe, source, 'edit', caption='scene',
                      schedule=Schedule('editing_t2i_editing_selective', 12, start, end),
                      diagnostics=diagnostics)
    handle.remove()
    assert len(calls) == 24
    assert pipe.transformer.attn_processors == processors
    assert diagnostics['attention'].shape == (4, 4)
    assert diagnostics['switched'] == (accept and end > 10)
    chosen = (Schedule(steps=12, start=max(10, start), end=end) if accept and end > 10
              else Schedule('pure_editing', 12))
    expected = generate(pipe, source, 'edit', caption='scene', schedule=chosen)
    np.testing.assert_array_equal(actual, expected)


def test_source_attention_matches_dense_reference():
    from switching.selective import source_attention
    torch.manual_seed(44)
    q, k = torch.randn(2, 13, 3, 8), torch.randn(2, 13, 3, 8)
    expected = (torch.einsum('qhd,khd->hqk', q[0, 3:8].bfloat16(), k[0].bfloat16())
                * (1 / 8 ** 0.5)).softmax(-1)[..., 8:13].sum(-1).mean(0).float().numpy()
    np.testing.assert_array_equal(source_attention(q, k, 3, 5, 5, chunk_size=2), expected)


def test_probe_processors_restore_on_error(pipe, monkeypatch):
    from switching.selective import AttentionProbe
    probe = AttentionProbe(16, 16, (4, 4))
    processors = dict(pipe.transformer.attn_processors)
    def fail(**kwargs):
        raise RuntimeError('deliberate')
    monkeypatch.setattr(pipe.transformer, 'forward', fail)
    with pytest.raises(RuntimeError, match='deliberate'):
        probe.forward(pipe.transformer, encoder_hidden_states=torch.zeros(1, 4, 24))
    assert pipe.transformer.attn_processors == processors


def test_selective_batched_cfg_matches_decision(pipe, source):
    encode = pipe.encode_prompt
    def padded(prompt, device=None, **kwargs):
        embeddings = encode(prompt, device=device)[0]
        embeddings = torch.nn.functional.pad(embeddings, (0, 0, 0, 4 - embeddings.shape[1]))
        return embeddings, pipe._prepare_text_ids(embeddings).to(device)
    pipe.encode_prompt = padded
    diagnostics = {}
    processors = dict(pipe.transformer.attn_processors)
    actual = generate(pipe, source, 'edit', caption='scene', sampling='paper',
                      schedule=Schedule('editing_t2i_editing_selective', 12, 0, 12),
                      diagnostics=diagnostics)
    assert pipe.transformer.attn_processors == processors
    assert np.isfinite(diagnostics['attention']).all()
    assert diagnostics['attention'].min() >= 0 and diagnostics['attention'].max() <= 1
    chosen = (Schedule(steps=12, start=10, end=12) if diagnostics['switched']
              else Schedule('pure_editing', 12))
    expected = generate(pipe, source, 'edit', caption='scene', sampling='paper', schedule=chosen)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('values,accepted', [([.1,.1,.1,.1], True),
                                            ([.3,.3,.3,.3], False),
                                            ([0,0,0,.2], False)])
def test_selective_statistics_and_rule(values, accepted):
    from switching.selective import AttentionProbe
    probe = AttentionProbe(4, 4, (2, 2))
    probe.maps = [np.array(values, dtype=np.float32)]
    chosen = probe.decide(Schedule('editing_t2i_editing_selective', 50, 10, 20))
    data = np.array(values, dtype=np.float32).astype(np.float64)
    assert probe.result['cv'] == pytest.approx(data.std() / (data.mean() + 1e-8))
    assert probe.result['pi'] == pytest.approx(np.percentile(data, 75))
    assert probe.result['switched'] == accepted
    assert chosen.mode == ('editing_t2i_editing' if accepted else 'pure_editing')


def test_9b_uses_native_sampler_and_execution_device(pipe, source):
    from switching.flux9b import generate as generate_9b
    expected = pipe(image=source, prompt='edit', num_inference_steps=4,
                    guidance_scale=4.0, generator=torch.Generator(device=pipe._execution_device).manual_seed(7)).images[0]
    actual = generate_9b(pipe, source, 'edit', schedule=Schedule('pure_editing', 4), seed=7)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize('sampling', ['diffusers', 'paper'])
def test_guidance_endpoints_match_switching(pipe, source, sampling):
    from switching.guidance import Guidance
    encode = pipe.encode_prompt
    def padded(prompt, device=None, **kwargs):
        emb = encode(prompt, device=device)[0]
        emb = torch.nn.functional.pad(emb, (0, 0, 0, 4 - emb.shape[1]))
        return emb, pipe._prepare_text_ids(emb).to(device)
    pipe.encode_prompt = padded
    base = dict(seed=7, sampling=sampling, caption='scene')
    pure = generate(pipe, source, 'edit', schedule=Schedule('pure_editing', 4), **base)
    switched = generate(pipe, source, 'edit', schedule=Schedule('editing_t2i_editing', 4, 1, 3), **base)
    for kind in ['image', 'multimodal', 'factorial']:
        actual = generate(pipe, source, 'edit', schedule=Schedule('pure_editing', 4),
                          guidance=Guidance(kind, alpha=1, start=1, end=3), **base)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(pure))
    for kind in ['multimodal', 'factorial']:
        actual = generate(pipe, source, 'edit', schedule=Schedule('pure_editing', 4),
                          guidance=Guidance(kind, alpha=0, w_i=0, w_ti=0, start=1, end=3), **base)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(switched))


@pytest.mark.parametrize('kind,expected', [('image', 17.25), ('multimodal', 11.25), ('factorial', 11.5)])
def test_guidance_formula_and_branch_conditions(kind, expected):
    from switching.guidance import Guidance, predict
    calls = []
    class Denoiser:
        dtype = torch.float32
        def __call__(self, hidden_states, encoder_hidden_states, **kw):
            source = hidden_states.shape[1] == 4
            text = int(encoder_hidden_states[0, 0, 0])
            calls.append((source, text))
            # U=1, T=3, I=2, E=6.
            value = {(False, 0): 1, (False, 2): 3, (True, 0): 2, (True, 1): 6}[source, text]
            return (torch.full_like(hidden_states, value),)
    pipe = types.SimpleNamespace(transformer=Denoiser())
    ids = torch.zeros(1, 2, 4)
    text = lambda value: (torch.full((1, 2, 3), value), ids)
    x = torch.ones(1, 2, 3)
    result = predict(pipe, x, ids, x, ids, {'instruction': text(1), 'caption': text(2)},
                     text(0), torch.tensor(500.), 4, Guidance(kind, alpha=.25, w_i=.5, w_ti=1))
    torch.testing.assert_close(result, torch.full_like(x, expected))
    assert len(calls) == (3 if kind == 'image' else 4)
    assert (True, 1) in calls and (True, 0) in calls and (False, 0) in calls
    assert ((False, 2) in calls) == (kind != 'image')


@pytest.mark.parametrize('kind,middle', [('image', [32, 32, 16]), ('multimodal', [32, 32, 16, 16])])
def test_guidance_interval_restores_editing(pipe, source, kind, middle):
    from switching.guidance import Guidance
    lengths = []
    hook = pipe.transformer.register_forward_pre_hook(
        lambda module, args, kwargs: lengths.append(kwargs['hidden_states'].shape[1]), with_kwargs=True)
    try:
        generate(pipe, source, 'edit', caption='scene', schedule=Schedule('pure_editing', 4),
                 guidance=Guidance(kind, alpha=.25, start=1, end=3))
    finally:
        hook.remove()
    assert lengths == [32, 32] + middle + middle + [32, 32]
