"""cwsteer — contrastive weight steering.

A weak-to-strong steering core extracted from w2schar-mini: fit ONE conditioned
adapter `y = h + c·δ` on contrastive (cho, rej) pairs, where δ is a banked
low-rank direction (LoRA) or the top-r SVD round-trip of W (PiSSA). c=0 is base,
+c/-c are signed poles of the SAME direction. Bake a chosen c into the weights
for inference, restore base when done.

This package is the stable core (adapter + train + bake). gen / filter /
calibrate / persona-search / eval land here as they are extracted (see
mft_honesty docs/spec/54_steering_lib.md).
"""

from cwsteer.adapter import LoRAConfig, ModulatedLoRA, ModulatedPiSSA
from cwsteer.bake import AdapterSpec, baked, pissa_to_lora_spec
from cwsteer.train import TrainCfg, train_adapter

__all__ = [
    "LoRAConfig",
    "ModulatedLoRA",
    "ModulatedPiSSA",
    "AdapterSpec",
    "baked",
    "pissa_to_lora_spec",
    "TrainCfg",
    "train_adapter",
]
