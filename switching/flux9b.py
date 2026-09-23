"""FLUX.2 Klein base 9B, using its native Diffusers sampling conventions."""

from . import flux

MODEL_ID = 'black-forest-labs/FLUX.2-klein-base-9B'
SELECTIVE_THRESHOLDS = (0.50, 0.209)


def load_pipeline(model=MODEL_ID, **kwargs):
    return flux.load_pipeline(model, **kwargs)


def generate(pipe, image, instruction, **kwargs):
    kwargs.setdefault('sampling', 'diffusers')
    kwargs.setdefault('generator_device', pipe._execution_device)
    kwargs.setdefault('selective_thresholds', SELECTIVE_THRESHOLDS)
    return flux.generate(pipe, image, instruction, **kwargs)
