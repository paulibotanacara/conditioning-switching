"""One conditioning choice per step, with zero-based, half-open intervals."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Conditioning:
    text: str
    source: bool


EDIT = Conditioning("instruction", True)
T2I = Conditioning("caption", False)
CAPTION_EDIT = Conditioning("caption", True)
IMPROVED_EDIT = Conditioning("improved_instruction", True)

# Conditioning outside and inside the interval
MODES = {
    "pure_editing": (EDIT, EDIT),
    "editing_t2i_editing": (EDIT, T2I),
    "editing_t2i_editing_selective": (EDIT, T2I),
    "editing_t2iep_editing": (EDIT, Conditioning("instruction", False)),
    "editing_i2ic_editing": (EDIT, CAPTION_EDIT),
    "editing_estar_editing": (EDIT, IMPROVED_EDIT),
    "i2ic_t2i_i2ic": (CAPTION_EDIT, T2I),
}


@dataclass(frozen=True)
class Schedule:
    mode: str = "editing_t2i_editing"
    steps: int = 50
    start: int = 10
    end: int = 16

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"Unknown mode {self.mode!r}; choose from {tuple(MODES)}")
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if self.mode == "pure_editing":
            return
        if type(self.start) is not int or not 0 <= self.start <= self.steps:
            raise ValueError("start must be an integer in [0, steps]")
        if type(self.end) is not int or not self.start <= self.end <= self.steps:
            raise ValueError("end must be an integer in [start, steps]")

    def at(self, step: int) -> Conditioning:
        if not 0 <= step < self.steps:
            raise IndexError(f"step must be in [0, {self.steps})")
        outer, middle = MODES[self.mode]
        return middle if self.start <= step < self.end else outer

    def required_texts(self) -> set[str]:
        texts = {self.at(k).text for k in range(self.steps)}
        if self.mode == "editing_t2i_editing_selective":
            texts.add("instruction")
        return texts
