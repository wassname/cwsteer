"""cwsteer — contrastive weight steering.

A weak-to-strong steering core extracted from w2schar-mini: fit ONE conditioned
adapter `y = h + c·δ` on contrastive (cho, rej) pairs, where δ is a banked
low-rank direction (LoRA) or the top-r SVD round-trip of W (PiSSA). c=0 is base,
+c/-c are signed poles of the SAME direction. Bake a chosen c into the weights
for inference, restore base when done.

The stable core is adapter + train + bake. Calibration (`c_scan`), on-policy pole
generation (`generate_pairs_from_personas`), the persona-leakage logit filter, and
probe replay (`run_probe`) are also here.
"""

from cwsteer.adapter import LoRAConfig, ModulatedLoRA, ModulatedPiSSA
from cwsteer.bake import AdapterSpec, baked, pissa_to_lora_spec
from cwsteer.train import TrainCfg, train_adapter
from cwsteer.c_scan import c_scan
from cwsteer.pairs import generate_pairs_from_personas, PersonaOnlyRepetitionPenalty
from cwsteer.dialogue import DialogueCfg, run_probe

__all__ = [
    "LoRAConfig",
    "ModulatedLoRA",
    "ModulatedPiSSA",
    "AdapterSpec",
    "baked",
    "pissa_to_lora_spec",
    "TrainCfg",
    "train_adapter",
    "c_scan",
    "generate_pairs_from_personas",
    "PersonaOnlyRepetitionPenalty",
    "DialogueCfg",
    "run_probe",
]
