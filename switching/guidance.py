"""Image and factorial multimodal guidance over a bounded denoising interval."""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class Guidance:
    kind: str = 'multimodal'
    alpha: float = 0.25
    start: int = 10
    end: int = 20
    w_i: float = 1.0
    w_ti: float = 4.0

    def validate(self, steps):
        if self.kind not in ('image', 'multimodal', 'factorial'):
            raise ValueError('Guidance kind must be image, multimodal, or factorial')
        if not all(math.isfinite(v) for v in (self.alpha, self.w_i, self.w_ti)):
            raise ValueError('Guidance weights must be finite')
        if not 0 <= self.alpha <= 1:
            raise ValueError('alpha must lie in [0,1]')
        if type(self.start) is not int or type(self.end) is not int or not 0 <= self.start <= self.end <= steps:
            raise ValueError('Require 0 <= start <= end <= steps')

    def coefficients(self, scale):
        if self.kind == 'image':
            return dict(u=1-self.alpha, t=0, i=self.alpha-scale, e=scale)
        wi, wti = ((self.alpha, self.alpha*scale) if self.kind == 'multimodal'
                   else (self.w_i, self.w_ti))
        return dict(u=1-scale-wi+wti, t=scale-wti, i=wi-wti, e=wti)

    def endpoint(self, scale):
        c = self.coefficients(scale)
        if c['u'] == c['t'] == 0:
            return 'editing'
        if c['i'] == c['e'] == 0:
            return 't2i'
        return None

    def needs_caption(self, scale):
        return self.start < self.end and self.coefficients(scale)['t'] != 0


def predict(pipe, latents, latent_ids, image_latents, image_ids, texts, negative,
            timestep, scale, guidance, paper=False):
    """Evaluate only nonzero corners; source tokens are absent for U and T."""
    coefficients = guidance.coefficients(scale)
    predictions = {}
    branches = dict(e=texts['instruction'], i=negative, u=negative)
    if 'caption' in texts:
        branches['t'] = texts['caption']
    for group, source in [('ei', True), ('tu', False)]:
        active = [b for b in group if coefficients[b] != 0]
        if not active:
            continue
        x, ids = latents, latent_ids
        if source:
            x, ids = torch.cat((x, image_latents), 1), torch.cat((ids, image_ids), 1)
        common = dict(hidden_states=x.to(pipe.transformer.dtype), img_ids=ids,
                      timestep=timestep.expand(len(latents)).to(pipe.transformer.dtype if paper else latents.dtype),
                      guidance=None, return_dict=False)
        if not paper:
            common['timestep'] = common['timestep'] / 1000
        if paper:
            count = len(active)
            common.update(hidden_states=common['hidden_states'].repeat(count, 1, 1),
                          img_ids=ids.repeat(count, 1, 1), timestep=common['timestep'].repeat(count))
            result = pipe.transformer(**common,
                encoder_hidden_states=torch.cat([branches[b][0] for b in active]),
                txt_ids=torch.cat([branches[b][1] for b in active]))[0][:, :latents.shape[1]]
            predictions.update(zip(active, result.chunk(count)))
        else:
            for b in active:
                predictions[b] = pipe.transformer(**common, encoder_hidden_states=branches[b][0],
                                                   txt_ids=branches[b][1])[0][:, :latents.shape[1]]
    # Accumulate in float32 to avoid cancellation in the factorial expansion.
    velocity = sum(coefficients[b] * value.float() for b, value in predictions.items())
    return velocity if paper else velocity.to(latents.dtype)
