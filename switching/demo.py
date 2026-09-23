"""Save a baseline and one selected mode using the same source and seed."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from .schedule import MODES, Schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--caption")
    parser.add_argument("--improved-instruction")
    parser.add_argument("--mode", choices=MODES, default="editing_t2i_editing")
    parser.add_argument("--start", type=int, default=10)
    parser.add_argument("--end", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling", choices=("diffusers", "paper"), default="diffusers")
    parser.add_argument("--negative-embeddings", type=Path,
                        help="Safetensors file containing the fixed null_embed tensor")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-base-4B")
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()
    schedule = Schedule(args.mode, args.steps, args.start, args.end)
    supplied = dict(instruction=args.instruction, caption=args.caption,
                    improved_instruction=args.improved_instruction)
    for key in schedule.required_texts() | {"instruction"}:
        if not supplied[key] or not supplied[key].strip():
            parser.error(f"--{key.replace('_', '-')} is required for this comparison")
    from .flux import generate, load_pipeline
    negative = None
    if args.negative_embeddings:
        from safetensors.torch import load_file
        negative = load_file(str(args.negative_embeddings))["null_embed"]

    with Image.open(args.image) as source:
        image = source.convert("RGB")
    pipe = load_pipeline(args.model, device=args.device, cpu_offload=args.cpu_offload)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, setting in (("baseline", Schedule("pure_editing", args.steps)),
                          (args.mode, schedule)):
        output = generate(pipe, image, args.instruction, caption=args.caption,
                          improved_instruction=args.improved_instruction,
                          schedule=setting, seed=args.seed, sampling=args.sampling,
                          negative_prompt_embeds=negative)
        output.save(args.output / f"{name}.png")
    metadata = dict(model=args.model, instruction=args.instruction, caption=args.caption,
                    improved_instruction=args.improved_instruction, seed=args.seed,
                    schedule=asdict(schedule), guidance_scale=4.0, sampling=args.sampling,
                    negative_embeddings=args.negative_embeddings.name if args.negative_embeddings else None)
    (args.output / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
