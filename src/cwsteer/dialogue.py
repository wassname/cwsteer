"""Probe replay: multi-turn greedy dialogue, byte-identical pre/post.

Each probe = `{id, opening, followups}`. We start a chat with the
opening, take the model's reply, then for each scripted followup append
it as the next user turn and take the next reply. Same probe set, same
followups, same decoding parameters pre and post → only the model's
replies differ.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from tqdm.auto import tqdm

from cwsteer.bake import AdapterSpec, baked


@dataclass
class DialogueCfg:
    max_new_tokens: int = 512
    enable_thinking: bool = False
    seed: int = 42


@torch.no_grad()
def _gen_one(model, tok, messages: list[dict], *, max_new_tokens: int,
             enable_thinking: bool, seed: int) -> str:
    rendered = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    enc = tok(rendered, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        # No eos_token_id override: use the model's own generation_config set.
        # gemma-4 ends a turn with <turn|> (id 106), not <eos> (id 1); its config
        # lists eos=[1,106,50]. Overriding with tok.eos_token_id=1 dropped 106, so
        # generation ran past the answer and looped empty <|channel>thought blocks
        # (the "thought"-loop collapse, task 28). Trusting the model's config is the
        # general fix (gemma-2/3, qwen all carry the right stops in their config).
    )
    cont = out[0, enc["input_ids"].shape[1]:]
    return tok.decode(cont, skip_special_tokens=True).strip()


def run_probe(model, tok, probe: dict, *, cfg: DialogueCfg) -> dict:
    """Run one probe: opening + each followup in sequence. Returns the
    full turn list."""
    turns: list[dict] = []
    messages = [{"role": "user", "content": probe["opening"]}]
    reply = _gen_one(model, tok, messages,
                     max_new_tokens=cfg.max_new_tokens,
                     enable_thinking=cfg.enable_thinking,
                     seed=cfg.seed)
    turns.append({"role": "user", "text": probe["opening"]})
    turns.append({"role": "assistant", "text": reply})
    messages.append({"role": "assistant", "content": reply})

    for fu in probe["followups"]:
        messages.append({"role": "user", "content": fu})
        reply = _gen_one(model, tok, messages,
                         max_new_tokens=cfg.max_new_tokens,
                         enable_thinking=cfg.enable_thinking,
                         seed=cfg.seed)
        turns.append({"role": "user", "text": fu})
        turns.append({"role": "assistant", "text": reply})
        messages.append({"role": "assistant", "content": reply})

    return {"id": probe["id"], "turns": turns}


def dialogue(model, tok, probes: list[dict], out_path: Path,
             *, hist_specs: Optional[list[AdapterSpec]] = None,
             current_spec: Optional[AdapterSpec] = None,
             c: float = 0.0,
             cfg: DialogueCfg = DialogueCfg()) -> dict:
    """Replay all probes under a `baked()` context combining history +
    optional current-round adapter. Writes JSON to `out_path`.

    - Pre-dialogue: pass `hist_specs` only, `current_spec=None`, `c` ignored.
    - Post-dialogue: pass `hist_specs` + `current_spec` + `c=signed_C`.

    `c` overrides current_spec.default_c (history uses each spec's
    own default_c, baked at their kept signed_C from calibration.json).
    """
    payload = {"id": out_path.stem, "c": c, "probes": []}
    hist_specs = hist_specs or []
    adapters = list(hist_specs)
    if current_spec is not None and c != 0.0:
        adapters.append(current_spec)
    cs = None
    if current_spec is not None and c != 0.0:
        cs = [s.default_c for s in hist_specs] + [c]

    desc = (f"dialogue @ c={c:+.3f}" if current_spec is not None and c != 0.0
            else f"dialogue @ c=0 (base+{len(hist_specs)} kept)")
    pbar = tqdm(probes, desc=desc, mininterval=10, leave=False)
    with baked(model, adapters, c_overrides=cs):
        for p in pbar:
            pbar.set_postfix_str(p["id"][:24])
            payload["probes"].append(run_probe(model, tok, p, cfg=cfg))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    logger.debug(f"dialogue → {out_path.name} ({len(probes)} probes @ c={c:+.3f})")
    _dump_dialogue(payload, out_path.stem, c)
    return payload


def _phase_tag(name: str) -> str:
    """interview_pre -> PRE, interview_post -> POST, else the stem upper-cased."""
    n = name.lower()
    if "post" in n:
        return "POST"
    if "pre" in n:
        return "PRE"
    return name.upper()


def _dump_dialogue(payload: dict, name: str, c: float) -> None:
    """Print the full probe dialogue to stdout, uncropped, between clear
    START/END banners so a log reader (human or LLM) can see exactly where the
    PRE vs POST interview begins and ends and where each probe splits. inspect-
    ai's tool-output view truncates to ~a line per turn, so the verbose / pueue
    log is the only place to read the whole reply, and POST collapse (gibberish,
    fused words, loops, language switches) hides mid-reply.
    """
    n = len(payload["probes"])
    tag = _phase_tag(name)
    out = [
        f"\n\n========== INTERVIEW {tag} @ c={c:+.3f} | {n} probes | uncropped ==========",
        "SHOULD: coherent first-person prose every turn. fused-words "
        "('understandinglives') / nonsense ('bago') / token-loops ('duck duck...') "
        "/ lang-switch => collapse at this c. Present at c>0 but absent at c=0 => "
        "signed_C too high for the multi-turn distribution.",
    ]
    for i, pr in enumerate(payload["probes"], 1):
        out.append(f"\n--- probe {i}/{n}: {pr['id']} ---")
        u = a = 0
        for t in pr["turns"]:
            if t["role"] == "user":
                out.append(f"[U{u}] {t['text']}"); u += 1
            else:
                out.append(f"[A{a}] {t['text']}"); a += 1
    out.append(f"\n========== END INTERVIEW {tag} ({n} probes) ==========\n")
    logger.info("\n".join(out))
