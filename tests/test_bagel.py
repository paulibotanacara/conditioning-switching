"""BAGEL schedule/CFG wiring and attention statistics, without model downloads."""
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from switching import bagel
from switching.schedule import Schedule, MODES, EDIT, T2I
from switching.policy import decide


@pytest.mark.parametrize('mode', [m for m in MODES if not m.endswith('selective')])
@pytest.mark.parametrize('bounds', [(1, 3), (0, 4), (0, 0), (4, 4)])
def test_context_switches_keep_one_latent(mode, bounds):
    schedule = Schedule(mode, 4, *bounds)
    conditions = {schedule.at(i) for i in range(4)}
    prepared = {c: dict(positive=c, negative_text=('image' if c.source else 'empty'),
                       negative_image=c.text) for c in conditions}
    calls = []
    def flow(**kw):
        calls.append(kw)
        return torch.ones_like(kw['x_t']) * (2 if kw['positive'].source else 3)
    pipe = SimpleNamespace(model=SimpleNamespace(_forward_flow=flow))
    initial = torch.ones(2, 3)
    actual = bagel.denoise(pipe, prepared, initial, schedule, guidance_scale=4,
                          image_guidance_scale=1.5, timestep_shift=3, cfg_renorm_min=1,
                          grid_shape=(1, 2), n_source=4, diagnostics={})
    times = torch.linspace(1, 0, 5)
    times = 3 * times / (1 + 2 * times)
    expected = initial.clone()
    assert len(calls) == 4
    for i, call in enumerate(calls):
        c = schedule.at(i)
        assert call['positive'] == c
        assert call['negative_text'] == ('image' if c.source else 'empty')
        assert call['negative_image'] == c.text
        assert call['cfg_renorm_type'] == ('text_channel' if c.source else 'global')
        assert call['cfg_text_scale'] == 4 and call['cfg_img_scale'] == 1.5
        torch.testing.assert_close(call['x_t'], expected)
        expected -= (2 if c.source else 3) * (times[i] - times[i+1])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('attention,expected_middle', [(0.2, T2I), (0.5, EDIT)])
def test_selective_probes_then_switches(monkeypatch, attention, expected_middle):
    calls = []
    @contextmanager
    def capture(model, indexes, n_source, maps):
        assert len(calls) == 9 and n_source == 7
        maps.extend([np.full(4, attention), np.full(4, attention)])
        yield
    monkeypatch.setattr(bagel, 'capture_attention', capture)
    def flow(**kw):
        calls.append(kw['condition'])
        return torch.zeros_like(kw['x_t'])
    model = SimpleNamespace(_forward_flow=flow, language_model=SimpleNamespace(model=SimpleNamespace(layers=[0, 1])))
    prepared = {c: dict(condition=c, packed_vae_token_indexes=torch.arange(4)) for c in (EDIT, T2I)}
    diagnostics = {}
    bagel.denoise(SimpleNamespace(model=model), prepared, torch.ones(4, 2),
                  Schedule('editing_t2i_editing_selective', 16, 0, 13),
                  guidance_scale=4, image_guidance_scale=1.5, timestep_shift=3,
                  cfg_renorm_min=1, grid_shape=(2, 2), n_source=7, diagnostics=diagnostics)
    assert calls == [EDIT]*10 + [expected_middle]*3 + [EDIT]*3
    assert diagnostics['thresholds'] == (0.166, 0.469)
    assert diagnostics['switched'] == (expected_middle == T2I)


def test_packed_gqa_attention_matches_dense_attention():
    torch.manual_seed(2)
    q, k = torch.randn(6, 4, 8), torch.randn(11, 2, 8)
    indexes = torch.tensor([1, 2, 4])
    actual = bagel.source_attention(q, k, indexes, 5, chunk_size=2)
    q, k = q[indexes].bfloat16(), k.repeat_interleave(2, dim=1).bfloat16()
    logits = torch.stack([q[:, h] @ k[:, h].T for h in range(4)])
    expected = (logits * (8**-0.5)).softmax(-1)[..., :5].sum(-1).mean(0).float().numpy()
    np.testing.assert_allclose(actual, expected)


def test_attention_patch_restored_on_failure(monkeypatch):
    module = SimpleNamespace(flash_attn_varlen_func=lambda *a: None)
    original = module.flash_attn_varlen_func
    monkeypatch.setattr(bagel.importlib, 'import_module', lambda name: module)
    model = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=[0])))
    with pytest.raises(RuntimeError):
        with bagel.capture_attention(model, torch.arange(1), 1, []):
            assert module.flash_attn_varlen_func is not original
            raise RuntimeError('forward failed')
    assert module.flash_attn_varlen_func is original


def test_model_specific_thresholds_change_decision():
    s = Schedule('editing_t2i_editing_selective', 50, 10, 16)
    attention = np.full((3, 3), 0.215)
    assert decide(attention, s, (0.50, 0.22))[1]['switched']
    assert not decide(attention, s, (0.50, 0.209))[1]['switched']
    assert decide(attention, s, bagel.SELECTIVE_THRESHOLDS)[1]['switched']
