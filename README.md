# Conditioning switching

Code for *When Text-to-Image Helps Editing: The Effects of Conditioning During Denoising*.

Use Python 3.10+ and CUDA-enabled PyTorch.

```bash
python -m pip install -e '.[flux,notebook]'
jupyter lab
```

| Model | Notebook |
| --- | --- |
| FLUX.2 Klein base 4B | [4B](examples/switching_4b.ipynb) |
| FLUX.2 Klein base 9B | [9B](examples/switching_9b.ipynb) |
| BAGEL | [BAGEL](examples/switching_bagel.ipynb) |

[Prompting with Qwen](examples/prompting.ipynb) generates a target caption and an improved editing instruction, then illustrates both with FLUX.2 Klein 4B.

The notebooks include generated examples, conditioning modes, and selective switching.
The 4B notebook also compares image and multimodal guidance.

Weights download from the official model repositories. For 9B, accept the
[access terms](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B)
and run `hf auth login`. BAGEL requires a separate environment; follow its notebook’s setup.

[Apache-2.0](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md).
