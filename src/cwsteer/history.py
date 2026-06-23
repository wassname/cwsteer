"""Load base model + kept-round adapters composed via HistoryBake.

Forked from `weight-steering-lite/src/wsl/load_with_history.py`, trimmed:
- Dropped 4-bit / BitsAndBytesConfig branch.
- Dropped flash-attention-2 hard requirement (set via env var if wanted).
- Single happy path: bf16, device_map="auto".

Round dirs whose `judgment.json.action == "keep"` count as history. The
caller resolves which dirs to compose (typically: all kept rounds before
this one, in order).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from cwsteer.adapter import HistoryBake, ModulatedLoRA, ModulatedPiSSA, PiSSAHistoryBake
from cwsteer.bake import AdapterSpec, adapter_spec_from_checkpoint


def _ckpt_kind(path: Path) -> str:
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as f:
        return f.metadata().get("kind", "lora")


_ROUND_RE = re.compile(r"^round(\d+)$")


def parse_round_n(name: str) -> int | None:
    m = _ROUND_RE.match(name)
    return int(m.group(1)) if m else None


def _load_base(model_id: str, dtype: torch.dtype, device_map: str,
               quant: str | None = None):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    quant_cfg = None
    if quant == "nf4":
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif quant is not None:
        raise ValueError(f"unknown quant {quant!r}; expected 'nf4' or None")

    attn_impl = os.environ.get("CSM_ATTN_IMPL", "eager")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map=device_map, torch_dtype=dtype,
        low_cpu_mem_usage=True, attn_implementation=attn_impl,
        quantization_config=quant_cfg,
    )
    model.eval()
    if quant_cfg is not None:
        logger.info(f"loaded {model_id} with nf4 quant (compute_dtype={dtype})")
    return model, tok


def load_base_with_history(
    model_id: str,
    history_dirs: Iterable[Path] | None = None,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    quant: str | None = None,
):
    """[TRAINING-ONLY] Load base + attach a gated `HistoryBake` hook for
    kept rounds. Returns (model, tokenizer, history_bake_or_None).

    Training code must call `history_bake.set_gate(lambda: lora._c != 0.0)`
    so the c=0 reference forward returns pristine base.

    For INFERENCE (eval / dialogue / gen) use `load_base_with_history_specs`
    instead — it returns CPU-resident AdapterSpec list for use with the
    `baked()` context manager (no hook overhead at forward).
    """
    model, tok = _load_base(model_id, dtype, device_map, quant=quant)
    history_dirs = list(history_dirs or [])
    if not history_dirs:
        return model, tok, None

    # Sniff kind from the first kept round — all rounds in a run share the
    # same adapter family (pipeline enforces this).
    first_kind = _ckpt_kind(history_dirs[0] / "adapter.safetensors")
    if any(_ckpt_kind(rd / "adapter.safetensors") != first_kind for rd in history_dirs):
        raise RuntimeError("mixed kinds across kept rounds — not supported")

    if first_kind == "pissa":
        # PiSSA: each load sequentially mutates layer.weight by extracting
        # that round's U·S·Vh. Order matters: round k's SVD was over the W
        # left by rounds 1..k-1, so we must load in the same order.
        adapters: list[tuple[ModulatedPiSSA, float]] = []
        for rd in history_dirs:
            adapter_path = rd / "adapter.safetensors"
            cal_path = rd / "calibration.json"
            if not adapter_path.exists():
                raise FileNotFoundError(f"kept-round {rd.name} missing adapter.safetensors")
            if not cal_path.exists():
                raise FileNotFoundError(f"kept-round {rd.name} missing calibration.json")
            signed_C = float(json.loads(cal_path.read_text())["signed_C"])
            adapter = ModulatedPiSSA.from_checkpoint(model, str(adapter_path))
            adapters.append((adapter, signed_C))
            logger.info(f"loaded {rd.name}/adapter(pissa) @ kept c={signed_C:+.4f}")
        history_bake = PiSSAHistoryBake(model, adapters)
        logger.info(f"loaded base + PiSSAHistoryBake over {len(adapters)} kept adapter(s)")
        return model, tok, history_bake

    history: list[tuple[ModulatedLoRA, float]] = []
    for rd in history_dirs:
        adapter_path = rd / "adapter.safetensors"
        cal_path = rd / "calibration.json"
        if not adapter_path.exists():
            raise FileNotFoundError(f"kept-round {rd.name} missing adapter.safetensors")
        if not cal_path.exists():
            raise FileNotFoundError(f"kept-round {rd.name} missing calibration.json")
        signed_C = float(json.loads(cal_path.read_text())["signed_C"])
        lora = ModulatedLoRA.from_checkpoint(model, str(adapter_path))
        history.append((lora, signed_C))
        logger.info(f"loaded {rd.name}/adapter @ kept c={signed_C:+.4f}")
    history_bake = HistoryBake(model, history)
    logger.info(f"loaded base + HistoryBake over {len(history)} kept adapter(s)")
    return model, tok, history_bake


def load_base_with_history_specs(
    model_id: str,
    history_dirs: Iterable[Path] | None = None,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    quant: str | None = None,
) -> tuple:
    """[INFERENCE] Load base; return kept-round adapters as CPU-resident
    AdapterSpec list (NO hooks attached). Caller uses
    `baked(model, hist_specs + [current_spec or []])` to apply them.

    Returns (model, tokenizer, hist_specs: list[AdapterSpec]).
    """
    model, tok = _load_base(model_id, dtype, device_map, quant=quant)
    hist_specs: list[AdapterSpec] = []
    for rd in (history_dirs or []):
        adapter_path = rd / "adapter.safetensors"
        cal_path = rd / "calibration.json"
        if not adapter_path.exists():
            raise FileNotFoundError(f"kept-round {rd.name} missing adapter.safetensors")
        if not cal_path.exists():
            raise FileNotFoundError(f"kept-round {rd.name} missing calibration.json")
        signed_C = float(json.loads(cal_path.read_text())["signed_C"])
        spec = adapter_spec_from_checkpoint(model, str(adapter_path), default_c=signed_C)
        hist_specs.append(spec)
        logger.info(f"loaded {rd.name}/adapter spec @ kept c={signed_C:+.4f}")
    if hist_specs:
        logger.info(f"loaded base + {len(hist_specs)} kept adapter spec(s) (no hooks)")
    return model, tok, hist_specs


def kept_history_dirs(slug_dir: Path, before_round: int | None = None) -> list[Path]:
    """Sorted list of `<slug_dir>/roundNN` paths whose judgment.action == 'keep'.
    If `before_round` is given, only include rounds with index < before_round.
    """
    keep = []
    for rd in sorted(p for p in slug_dir.glob("round*") if p.is_dir()):
        n = parse_round_n(rd.name)
        if n is None:
            continue
        if before_round is not None and n >= before_round:
            continue
        j = rd / "judgment.json"
        if j.exists() and json.loads(j.read_text()).get("action") == "keep":
            keep.append(rd)
    return keep
