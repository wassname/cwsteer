"""Path-loss ModulatedLoRA training with reverse-KL coherence penalty.

Forked from `weight-steering-lite/src/wsl/train.py`, trimmed:
- Full KL on logits (no top-K approximation). Fine for the smaller models
  + smaller batches we use here; swap to top-K if memory becomes the
  bottleneck.
- Dropped held-out probe-eval loop (we evaluate via dialogue after train).

Per (prompt, cho, rej), sampled C ∈ (0, 1]:

    nll̃_b(x) ≡ nll_b(x) / max(detach(nll_b(x)), 1)     (PUSH side only)
    L_pos_nll = C · (mean_b nll(cho|+C) − mean_b nll̃(rej|+C))
    L_neg_nll = C · (mean_b nll(rej|−C) − mean_b nll̃(cho|−C))

Margin formulation: subtraction cancels the shared-fluency direction
(both cho and rej would lower nll under a fluency-only adapter), so
the surviving gradient component is the persona axis.

**Asymmetric cap — PUSH only.** Cross-entropy is asymmetric: the PULL
term (minimize nll toward the label) has a bounded, self-limiting
gradient that → 0 as nll → 0; the PUSH term (maximize nll away from
the label) grows without bound (∇(-log p) ∝ 1/p) and under raw margin
dominates θ-updates, blowing up to nll ~ 60 with ‖g‖ ~ 5800 (task 101).
So we cap ONLY the two PUSH terms via `_normed_mean` (divide each
sample by its detached nll, floored at 1 → ‖per-sample grad‖ scale-
free, runaway tamed) and leave the two PULL terms at raw `.mean()`.

Why PULL stays raw: capping it (task 101/20 capped BOTH) inverted the
intended balance. The on-policy rej-pull (nll < 1, floored → unscaled)
ran at full gradient and learned from step 1, while the off-policy
cho-pull (nll ~ 3 → throttled ×1/3) — the actual behaviour change —
stayed stuck. Raw PULL lets the far pole (cho) make the bigger
gradient, which is the direction we want to learn.

Replaces the post-autograd per-side `TARGET_G_NORM=5.0` equalization
in 6e1ec5c, which equalized BETWEEN the two C-sign frames but not
WITHIN each frame (cho-pull vs rej-push); the within-frame asymmetry
was the actual runaway source.

β trades steering signal vs distribution shift; weight-decay + β
jointly pull toward "no-op" while NLL pulls toward cho/rej.

`base` is the c=0 forward = pristine base (HistoryBake gate disabled
when `lora._c == 0`). So a new adapter that fights a prior bake pays
its KL bill cumulatively from base, not iteratively from last bake.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from cwsteer.adapter import ModulatedLoRA, ModulatedPiSSA


@dataclass
class TrainCfg:
    r: int = 16
    alpha: float = 32.0
    targets: tuple[str, ...] = ("all-linear",)
    layer_range: tuple[float, float] = (0.0, 1.0)
    steps: int = 200
    warmup_ratio: float = 0.1
    batch_size: int = 4
    lr: float = 2e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    max_len: int = 512
    n_val_pairs: int = 3
    min_val_improvement: float = 0.05
    kl_lambda: float = 0.032
    """β: coefficient on mean reverse-KL per step (nats, matches NLL units).
    0 disables. Bump up if Δnll blows past +0.02 (coherence breaks); bump
    down if eval Δ stays at noise."""
    pcgrad: bool = True
    """ON (2026-06-06, re-enabled). Was OFF 2026-06-03 as near-inert on the
    margin loss (task-31: conflict branch fired 42/240 steps, cos median 0.027 —
    the margin formulation already killed the old cos≈-0.97 shared-fluency
    conflict). Re-enabled because the per-pair EDIT workflow can push cho
    off-policy relative to its on-policy rej, reintroducing the cho-pull vs
    rej-push gradient conflict PCGrad de-conflicts (audit lens e). Cheap: the
    projection only fires on steps where cos<0, and cos/conflict are logged
    either way, so if a run shows it still never fires we drop it again."""
    seed: int = 42
    # ─ PiSSA-only ─ (ignored for ModulatedLoRA)
    pissa_selection_score: str = "cho_rej_min_std"
    """One of {"s_only", "wanda", "act_only", "cho_rej_min_std"}. The default
    picks directions that are alive in BOTH cho and rej completion-token
    activations — see ModulatedPiSSA docstring."""
    pissa_calib_max_tokens: int = 1024
    """Cap captured tokens per layer for activation-driven top-r selection."""


# ---------------------------------------------------------------------------
# Tokenisation: prompt+completion teacher-forced; label mask = prompt -100.
# Persona prefix is DROPPED at train time so the adapter learns the
# behaviour conditioned only on c, not on the persona prefix.
# ---------------------------------------------------------------------------

def build_tokens(tok, prompt: str, completion: str, max_len: int,
                 *, enable_thinking: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """(input_ids, labels) where labels mask out the prompt portion."""
    prompt_text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    full_text = prompt_text + completion + tok.eos_token
    prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
    full_ids = tok(full_text, add_special_tokens=False).input_ids
    full_ids = full_ids[:max_len]
    labels = list(full_ids)
    for i in range(min(len(prompt_ids), len(labels))):
        labels[i] = -100
    return torch.tensor(full_ids), torch.tensor(labels)


def collate(batch: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int):
    max_len = max(b[0].shape[0] for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, (ids, lbl) in enumerate(batch):
        n = ids.shape[0]
        input_ids[i, :n] = ids
        labels[i, :n] = lbl
        attn[i, :n] = 1
    return input_ids, labels, attn


class PairDataset(Dataset):
    def __init__(self, pairs: list[dict], tok, max_len: int,
                 enable_thinking: bool = False):
        self.pairs = pairs
        self.tok = tok
        self.max_len = max_len
        self.enable_thinking = enable_thinking

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p = self.pairs[i]
        ids_p, lbl_p = build_tokens(self.tok, p["prompt"], p["cho"], self.max_len,
                                    enable_thinking=self.enable_thinking)
        ids_n, lbl_n = build_tokens(self.tok, p["prompt"], p["rej"], self.max_len,
                                    enable_thinking=self.enable_thinking)
        return (ids_p, lbl_p), (ids_n, lbl_n)


def _pair_collate(batch, pad_id):
    pos = [b[0] for b in batch]
    neg = [b[1] for b in batch]
    ip, lp, ap = collate(pos, pad_id)
    in_, ln, an = collate(neg, pad_id)
    return ip, lp, ap, in_, ln, an


def _zerofill(grads, params):
    return [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]


def _per_sample_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-sample mean NLL over completion (label != -100) tokens. Matches
    HF causal-LM reduction (logits[t] predicts label[t+1]) but un-reduced
    over batch. Returns shape [B]."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_per_tok = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none", ignore_index=-100,
    ).view(shift_labels.size())
    mask = (shift_labels != -100).float()
    return (loss_per_tok * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)


def _normed_mean(nll_b: torch.Tensor) -> torch.Tensor:
    """nll_b / max(detach(nll_b), 1), then batch-averaged. PUSH-side only:
    caps each sample's per-step gradient at ~unit by cancelling the 1/p
    blow-up of CE's away-from-label (maximize-nll) side. Floor at 1 means
    samples already below nll=1 pass through unscaled. PULL terms do NOT
    use this — see module docstring (capping pull throttled the off-policy
    cho-pull and inverted the intended balance)."""
    denom = nll_b.detach().clamp_min(1.0)
    return (nll_b / denom).mean()


@torch.no_grad()
def _val_nll(model, lora, val_pairs: list[dict], tok, max_len: int,
             *, C: float, enable_thinking: bool) -> tuple[float, float]:
    """Held-out nll(cho|+C), nll(rej|-C): the val analogues of the train
    nll+/nll- columns, on pairs the adapter never trained on. cho is the
    teacher's OFF-policy edit, so val nll+ falling means the adapter LEARNS
    the target direction rather than memorizing the train cho — the one
    signal that distinguishes a converged fit from an overfit one (train
    nll alone cannot)."""
    ds = PairDataset(val_pairs, tok, max_len, enable_thinking=enable_thinking)
    ip, lp, ap, in_, ln, an = (
        t.to(next(model.parameters()).device)
        for t in _pair_collate([ds[i] for i in range(len(ds))], tok.pad_token_id))
    was_training = model.training
    model.eval()
    # no_grad: val is a pure diagnostic, gradients are never used. Without it
    # model.eval() still builds the full autograd graph (eval only toggles
    # dropout/bn), and this collates the WHOLE val set in one full-vocab forward
    # — at 31b that graph alone OOMs the next train step (task 39).
    with torch.no_grad():
        with lora(model, c=+C):
            nll_cho = _per_sample_nll(model(input_ids=ip, attention_mask=ap).logits.float(), lp).mean()
        with lora(model, c=-C):
            nll_rej = _per_sample_nll(model(input_ids=in_, attention_mask=an).logits.float(), ln).mean()
    if was_training:
        model.train()
    return float(nll_cho), float(nll_rej)


_KL_TOPK = 256


def _kl_topk_base(logits_steer, base_top_logp_K, base_top_idx, labels):
    """Reverse KL(p_steer_K ‖ p_base_K) on the top-K simplex per position,
    where p_*_K are the steered/base distributions renormalized over
    base's top-K indices. HF-style shift (logits[t] predicts label[t+1]).

    Memory fix: takes raw `logits_steer` (B, S, V), NOT full log_softmax,
    because logsumexp_full cancels in the K-renorm:
        log_softmax(logits)[K] - logsumexp(log_softmax(logits)[K])
      = logits[K] - logsumexp_full - (logsumexp(logits[K]) - logsumexp_full)
      = logits[K] - logsumexp(logits[K])
    so we gather raw logits at base_top_idx and skip the full-vocab
    log_softmax allocation entirely (was 8 GiB / forward × 2 = 16 GiB
    retained in autograd graph at gemma-2b batch=8 seq=1024).
    `base_top_logp_K` is already K-renormalized (computed once at the
    no-grad base forward).

    Renormalization is REQUIRED for KL ≥ 0: the K-subset sums to <1
    under full-V softmax; naïve Σ p_s · (logp_s − logp_b) on those is
    unbounded in sign (anti-anchor pathology in earlier trace).
    Gibbs ≥ 0 holds after both sides renormalize over K.

    Approximation bias: discards mass outside base's top-K (no outside
    bucket). For an instruct LM at K=256 base captures >99% per position.
    """
    s_sh = logits_steer[:, :-1, :]                     # (B, S-1, V) raw logits
    b_sh_logp_K = base_top_logp_K[:, :-1, :]           # (B, S-1, K) K-renormed
    b_sh_idx = base_top_idx[:, :-1, :]                 # (B, S-1, K)
    mask_sh = (labels[:, 1:] != -100)
    s_top = torch.gather(s_sh, -1, b_sh_idx).float()   # (B, S-1, K) fp32 at reduction
    logp_s_K = s_top - torch.logsumexp(s_top, dim=-1, keepdim=True)
    p_s_K = logp_s_K.exp()
    kl_per_tok = (p_s_K * (logp_s_K - b_sh_logp_K)).sum(dim=-1)
    return kl_per_tok[mask_sh.bool()].mean()


def pcgrad_train_step(
    model, lora: ModulatedLoRA,
    ip, lp, ap, in_, ln, an,
    params: list,
    *,
    C: float,
    pcgrad: bool = True,
    kl_lambda: float = 0.0,
) -> dict:
    """One step: margin NLL on both poles + (optional) KL anchor to c=0.
    PCGrad operates on the (margin_pos, margin_neg) gradients only — KL
    is added unprojected.

    Margin formulation:
      L_pos = C · (mean nll(cho|+C) − mean nll(rej|+C))
      L_neg = C · (mean nll(rej|−C) − mean nll(cho|−C))
    The shared-fluency direction (cho and rej both lower nll when adapter
    fits style) cancels in the difference; only the persona-axis component
    survives. Earlier per-sample-NLL formulation was dominated by the shared
    component (cos(g_pos, g_neg) ≈ -0.97 just from the c-sign flip on a
    shared signal).
    """
    use_kl = kl_lambda > 0
    device = next(p.device for p in params)
    zero = torch.zeros((), device=device)

    # ---- c=0 reference forwards (no grad, raw logits → top-K → K-renorm) ----
    # Skip the full-vocab log_softmax: topk on raw logits gives the same
    # indices (softmax is monotonic), and the K-renormalized log-prob is
    # what _kl_topk_base actually needs. Memory: avoids the 8 GiB full
    # log_softmax allocation. fp32 at the K-renorm logsumexp is cheap.
    if use_kl:
        with torch.no_grad():
            with lora(model, c=0.0):
                logits_b_p = model(input_ids=ip, attention_mask=ap).logits
                b_top_p = logits_b_p.topk(_KL_TOPK, dim=-1)
                base_top_idx_p = b_top_p.indices
                v_p = b_top_p.values.float()
                base_top_logp_p = v_p - torch.logsumexp(v_p, dim=-1, keepdim=True)
                del logits_b_p, b_top_p, v_p
                logits_b_n = model(input_ids=in_, attention_mask=an).logits
                b_top_n = logits_b_n.topk(_KL_TOPK, dim=-1)
                base_top_idx_n = b_top_n.indices
                v_n = b_top_n.values.float()
                base_top_logp_n = v_n - torch.logsumexp(v_n, dim=-1, keepdim=True)
                del logits_b_n, b_top_n, v_n

    # ---- both inputs at c=+C  (cho prefer, rej penalize) ------------------
    with lora(model, c=+C):
        out_cp = model(input_ids=ip, attention_mask=ap)
        nll_cho_p_b = _per_sample_nll(out_cp.logits.float(), lp)
        out_rp = model(input_ids=in_, attention_mask=an)
        nll_rej_p_b = _per_sample_nll(out_rp.logits.float(), ln)
        # PULL (cho toward labels): raw mean — CE pull is bounded and self-
        # limits as nll→0, so it needs no cap; full gradient lets the
        # off-policy cho (far, nll~3) DRIVE the behaviour change instead of
        # being throttled to 1/nll (task 101/20: capping both inverted the
        # dominance — on-policy rej-pull learned from step 1, cho-pull stuck).
        # PUSH (rej away from labels): keep the 1/nll cap — maximizing nll is
        # the unbounded ∇(-log p)∝1/p runaway _normed_mean was built to tame.
        L_pos_nll = C * (nll_cho_p_b.mean() - _normed_mean(nll_rej_p_b))
        mean_nll_p = nll_cho_p_b.mean().detach().item()
        if use_kl:
            kl_p = _kl_topk_base(out_cp.logits, base_top_logp_p, base_top_idx_p, lp)
            L_pos_kl = kl_lambda * kl_p
    g_pos_nll = _zerofill(torch.autograd.grad(
        L_pos_nll, params, retain_graph=use_kl, allow_unused=True), params)
    if use_kl:
        g_pos_kl = _zerofill(torch.autograd.grad(
            L_pos_kl, params, retain_graph=False, allow_unused=True), params)
    else:
        g_pos_kl = [torch.zeros_like(p) for p in params]
        kl_p = zero

    # ---- both inputs at c=-C  (rej prefer, cho penalize) ------------------
    with lora(model, c=-C):
        out_rn = model(input_ids=in_, attention_mask=an)
        nll_rej_n_b = _per_sample_nll(out_rn.logits.float(), ln)
        out_cn = model(input_ids=ip, attention_mask=ap)
        nll_cho_n_b = _per_sample_nll(out_cn.logits.float(), lp)
        # PULL (rej toward labels) raw; PUSH (cho away) capped — see L_pos.
        L_neg_nll = C * (nll_rej_n_b.mean() - _normed_mean(nll_cho_n_b))
        mean_nll_n = nll_rej_n_b.mean().detach().item()
        if use_kl:
            kl_n = _kl_topk_base(out_rn.logits, base_top_logp_n, base_top_idx_n, ln)
            L_neg_kl = kl_lambda * kl_n
    g_neg_nll = _zerofill(torch.autograd.grad(
        L_neg_nll, params, retain_graph=use_kl, allow_unused=True), params)
    if use_kl:
        g_neg_kl = _zerofill(torch.autograd.grad(
            L_neg_kl, params, retain_graph=False, allow_unused=True), params)
    else:
        g_neg_kl = [torch.zeros_like(p) for p in params]
        kl_n = zero

    # ---- PCGrad on the NLL pair --------------------------------------------
    # Within-side asymmetry handled at the loss via _normed_mean (above);
    # nothing extra at the grad level here.
    gp_flat = torch.cat([g.reshape(-1) for g in g_pos_nll])
    gn_flat = torch.cat([g.reshape(-1) for g in g_neg_nll])
    dot = (gp_flat * gn_flat).sum()
    gp_norm_sq = (gp_flat * gp_flat).sum().clamp_min(1e-12)
    gn_norm_sq = (gn_flat * gn_flat).sum().clamp_min(1e-12)
    cos = (dot / (gp_norm_sq.sqrt() * gn_norm_sq.sqrt())).item()
    conflict = dot.item() < 0

    if pcgrad and conflict:
        gp_proj = gp_flat - (dot / gn_norm_sq) * gn_flat
        gn_proj = gn_flat - (dot / gp_norm_sq) * gp_flat
        nll_summed = 0.5 * (gp_proj + gn_proj)
    else:
        nll_summed = 0.5 * (gp_flat + gn_flat)

    if use_kl:
        kl_flat = 0.5 * (
            torch.cat([g.reshape(-1) for g in g_pos_kl]) +
            torch.cat([g.reshape(-1) for g in g_neg_kl])
        )
        summed = nll_summed + kl_flat
        g_kl_norm = float(kl_flat.norm())
    else:
        summed = nll_summed
        g_kl_norm = 0.0
    g_nll_norm = float(nll_summed.norm())

    offset = 0
    for p in params:
        n = p.numel()
        p.grad = summed[offset:offset + n].view_as(p)
        offset += n

    return {
        "L_pos_nll": mean_nll_p,           # raw mean NLL (not normalized) for interpretability
        "L_neg_nll": mean_nll_n,
        "kl_mean_pos": kl_p.detach().item() if use_kl else 0.0,
        "kl_mean_neg": kl_n.detach().item() if use_kl else 0.0,
        "g_nll_norm": g_nll_norm,          # ‖summed NLL gradient‖ (post-PCGrad, pre-clip)
        "g_kl_norm": g_kl_norm,            # ‖summed KL gradient‖   (post-PCGrad, pre-clip)
        "C": C,
        "conflict": conflict,
        "cos": cos,
    }


def _capture_calibration_activations(model, tok, pairs: list[dict],
                                     cfg: TrainCfg, *,
                                     enable_thinking: bool
                                     ) -> dict[str, dict[str, torch.Tensor]]:
    """Capture per-layer input activations on COMPLETION tokens during full
    (prompt+cho) and (prompt+rej) forwards. Returns
    `dict[layer_name, {"cho": (N_cho, d_in), "rej": (N_rej, d_in)}]` (CPU
    float32, capped per side at cfg.pissa_calib_max_tokens).

    Why completion-only: the persona axis lives in what the model PRODUCES;
    prompt-token activations are identical between cho and rej (same prompt)
    so a contrast on them yields zero signal. Masking to completion tokens
    (labels != -100) isolates the cho/rej-specific activations.

    SHOULD: cho and rej captures produce DIFFERENT distributions per layer.
    ELSE selection score `cho_rej_min_std` collapses to S-bias and we lose
    the contrast (the whole point of the fork).
    """
    from cwsteer.adapter import LoRAConfig, _find_targets

    probe_cfg = LoRAConfig(r=cfg.r, alpha=1.0, targets=cfg.targets,
                           layer_range=cfg.layer_range, dtype=next(model.parameters()).dtype)
    targets = _find_targets(model, probe_cfg)
    cap = cfg.pissa_calib_max_tokens

    # Per-side state; reset between cho and rej passes.
    captured: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {"cho": [], "rej": []} for name, _ in targets}
    counts: dict[str, dict[str, int]] = {
        name: {"cho": 0, "rej": 0} for name, _ in targets}
    _state: dict[str, object] = {"side": "cho", "completion_mask": None}

    def make_hook(name):
        def _h(module, args, kwargs):
            side: str = _state["side"]
            if counts[name][side] >= cap:
                return
            x = args[0].detach()
            mask = _state["completion_mask"]
            if mask is not None and mask.shape[:x.dim() - 1] == x.shape[:-1]:
                x = x[mask.bool()]                        # (N_real, d_in)
            else:
                x = x.reshape(-1, x.shape[-1])
            x = x.to(torch.float32).cpu()
            take = min(x.shape[0], cap - counts[name][side])
            captured[name][side].append(x[:take])
            counts[name][side] += take
        return _h

    handles = [layer.register_forward_pre_hook(make_hook(name), with_kwargs=True)
               for name, layer in targets]
    try:
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        ds = PairDataset(pairs, tok, cfg.max_len, enable_thinking=enable_thinking)
        bs = max(1, cfg.batch_size)
        for side in ("cho", "rej"):
            _state["side"] = side
            # Build tensors per side from PairDataset (already builds full
            # prompt+completion ids with labels masking the prompt to -100).
            samples = [ds[i][0 if side == "cho" else 1] for i in range(len(ds))]
            with torch.no_grad():
                for i in range(0, len(samples), bs):
                    if all(counts[n][side] >= cap for n in counts):
                        break
                    batch = samples[i:i+bs]
                    ids, lbl, attn = collate(batch, tok.pad_token_id)
                    ids, lbl, attn = ids.to(device), lbl.to(device), attn.to(device)
                    # Only label != -100 positions = completion tokens.
                    _state["completion_mask"] = (lbl != -100)
                    model(input_ids=ids, attention_mask=attn)
            _state["completion_mask"] = None
        if was_training:
            model.train()
    finally:
        for h in handles:
            h.remove()

    X: dict[str, dict[str, torch.Tensor]] = {}
    for name, sides in captured.items():
        X[name] = {
            s: torch.cat(chunks, dim=0) if chunks else torch.empty(0)
            for s, chunks in sides.items()
        }
    n_cho = [v["cho"].shape[0] for v in X.values() if v["cho"].numel() > 0]
    n_rej = [v["rej"].shape[0] for v in X.values() if v["rej"].numel() > 0]
    logger.info(
        f"PiSSA calibration: captured {len(X)} targets, "
        f"cho tokens/layer in [{min(n_cho, default=0)}, {max(n_cho, default=0)}], "
        f"rej tokens/layer in [{min(n_rej, default=0)}, {max(n_rej, default=0)}] "
        f"(cap={cap})"
    )
    return X


def train_adapter(model, tok, pairs: list[dict], cfg: TrainCfg,
                  *, history_bake=None, enable_thinking: bool = False,
                  adapter_cls: type = ModulatedLoRA):
    """Fit one ModulatedLoRA / ModulatedPiSSA on `pairs` via path-loss + KL anchor.

    `history_bake`: if given, its gate is set to `lambda: lora._c != 0.0` so
    the c=0 reference returns pristine base (LoRA) or post-prior-bakes
    (PiSSA — `PiSSAHistoryBake.set_gate` is a no-op; see adapter.py).
    `adapter_cls`: ModulatedLoRA (default) or ModulatedPiSSA.
    """
    torch.manual_seed(cfg.seed)

    # Hold out a small explicit val slice as the overfit canary. The user asked
    # for fail-fast research behavior here: if the adapter does not lower held-
    # out val nll+, the round should die loudly instead of "training" on paper.
    perm = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    n_val = min(cfg.n_val_pairs, len(pairs) // 4)
    val_pairs = [pairs[i] for i in perm[:n_val]]
    pairs = [pairs[i] for i in perm[n_val:]]

    if adapter_cls is ModulatedPiSSA:
        calib = _capture_calibration_activations(
            model, tok, pairs, cfg, enable_thinking=enable_thinking,
        )
        lora = ModulatedPiSSA(model, r=cfg.r, targets=cfg.targets,
                              layer_range=cfg.layer_range,
                              dtype=next(model.parameters()).dtype,
                              calibration_activations=calib,
                              selection_score=cfg.pissa_selection_score)
    elif adapter_cls is ModulatedLoRA:
        lora = ModulatedLoRA(model, r=cfg.r, alpha=cfg.alpha, targets=cfg.targets,
                             layer_range=cfg.layer_range,
                             dtype=next(model.parameters()).dtype)
    else:
        raise ValueError(f"unknown adapter_cls={adapter_cls!r}")
    params = list(lora.parameters())
    optim = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=int(cfg.warmup_ratio * cfg.steps),
        num_training_steps=cfg.steps,
    )
    if history_bake is not None:
        history_bake.set_gate(lambda: lora._c != 0.0)

    ds = PairDataset(pairs, tok, cfg.max_len, enable_thinking=enable_thinking)
    pad_id = tok.pad_token_id
    # drop_last=False: small training pools (e.g. n_train_pairs=15,
    # batch_size=16) would yield zero full batches with drop_last=True and
    # raise StopIteration on the very first step — masked through inspect-ai
    # as a vague "train_student tool unresponsive". _per_sample_nll handles
    # ragged batches fine; nothing in the loss depends on a fixed batch axis.
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=lambda b: _pair_collate(b, pad_id),
        drop_last=False,
    )

    device = next(model.parameters()).device
    it = iter(loader)
    pbar = tqdm(range(cfg.steps), desc="train", leave=False)
    traces: list[dict] = []
    val_traces: list[dict] = []
    _VAL_EVERY = 30
    _VAL_PATIENCE = 3   # stop after 3 val-checks (=90 steps) with no val-nll+ gain
    warmup_steps = max(1, int(cfg.warmup_ratio * cfg.steps))
    # Early-stop = deploy the weights at the val-nll+ MINIMUM, AND actually halt
    # once val-nll+ stops improving — no point spending compute past the min only
    # to memorize the train pairs (gemma task-31 val+ 2.76→2.89 flat then
    # detonates; task-38 1.74→10.2). val_nll+ is the +C pole we bake/deploy. We
    # gate BOTH the best-state snapshot and the patience break on warmup (lr has
    # peaked): pre-warmup the adapter is still near its B=0 init, so a step-0
    # val-min would otherwise deploy a ~null adapter (gemma-3-4b run: several kept
    # rounds banked best_step=0 = no intervention; RJ 2026-06-16). warmup_steps>=1
    # so even a tiny step budget excludes the untrained snapshot.
    best_val_pos = float("inf")
    best_state: list[torch.Tensor] | None = None
    stale = 0           # consecutive post-warmup val-checks with no improvement
    for step in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        ip, lp, ap, in_, ln, an = (t.to(device) for t in batch)

        # C fixed at 1.0 (was U[0.5, 2] per step). We now bake at the trained
        # strength (init_c=1.0, backoff removed in c_scan), so train-c and
        # inference-c coincide — no need to spread training over a range to keep
        # the bake point in-distribution. Pinning concentrates the signal at the
        # one strength we deploy. KL anchor still handles "be quiet at low c".
        C = 1.0
        trace = pcgrad_train_step(
            model, lora, ip, lp, ap, in_, ln, an, params,
            C=C, pcgrad=cfg.pcgrad, kl_lambda=cfg.kl_lambda,
        )

        gn_pre = float(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip))
        optim.step()
        optim.zero_grad(set_to_none=True)
        sched.step()

        lr = optim.param_groups[0]["lr"]
        with torch.no_grad():
            ds_norm = float(torch.stack(
                [p.detach().float().norm() for p in params]).mean())
        traces.append({
            "step": step,
            "C": trace["C"],
            "nll+": trace["L_pos_nll"],
            "nll-": trace["L_neg_nll"],
            "kl+": trace["kl_mean_pos"],
            "kl-": trace["kl_mean_neg"],
            "cos": trace["cos"],
            "‖Δs‖": ds_norm,
            "‖g_nll‖": trace["g_nll_norm"],
            "‖g_kl‖": trace["g_kl_norm"],
            "‖g‖": gn_pre,
            "lr": lr,
            "conf": int(trace["conflict"]),
        })

        if val_pairs and (step % _VAL_EVERY == 0 or step == cfg.steps - 1):
            v_cho, v_rej = _val_nll(model, lora, val_pairs, tok, cfg.max_len,
                                    C=C, enable_thinking=enable_thinking)
            val_traces.append({"step": step,
                               "train_nll+": trace["L_pos_nll"], "val_nll+": v_cho,
                               "train_nll-": trace["L_neg_nll"], "val_nll-": v_rej})
            if v_cho < best_val_pos and step >= warmup_steps:
                best_val_pos = v_cho
                best_state = [p.detach().clone() for p in params]
                stale = 0
            elif step >= warmup_steps:
                stale += 1
                if stale >= _VAL_PATIENCE:
                    logger.info(
                        f"early-stop: val-nll+ no improvement for {stale} checks "
                        f"(best={best_val_pos:.3g} at step "
                        f"{min(val_traces, key=lambda t: t['val_nll+'])['step']}); "
                        f"halting at step {step}/{cfg.steps - 1}.")
                    break

    if history_bake is not None:
        history_bake.set_gate(lambda: True)              # restore inference default

    # Restore the val-nll+ minimum (overfit fix). No val (tiny pool) → keep last.
    best_step = None
    step0_val_pos = None
    if val_traces:
        # best_step must match the DEPLOYED weights: only post-warmup checks are
        # eligible for best_state, so rank over the same window (else a step-0
        # global min would mislabel the deploy).
        eligible = [t for t in val_traces if t["step"] >= warmup_steps] or val_traces
        best_step = min(eligible, key=lambda t: t["val_nll+"])["step"]
        step0_val_pos = val_traces[0]["val_nll+"]
    if best_state is not None:
        last_step = val_traces[-1]["step"]
        with torch.no_grad():
            for p, b in zip(params, best_state):
                p.copy_(b)
        logger.info(
            f"early-stop: deploying adapter at val-nll+ min "
            f"(step {best_step}, val+={best_val_pos:.3g}) vs last step {last_step} "
            f"(val+={val_traces[-1]['val_nll+']:.3g}). "
            f"best==last → no overfit; best≪last → memorization avoided."
        )

    _log_train_table(traces)
    _log_val_table(val_traces)
    improvement = None
    if step0_val_pos is not None:
        improvement = step0_val_pos - best_val_pos
    lora.train_summary = {
        "n_train_pairs": len(pairs),
        "n_val_pairs": len(val_pairs),
        "best_step": best_step,
        "best_val_nll_pos": None if best_val_pos == float("inf") else best_val_pos,
        "step0_val_nll_pos": step0_val_pos,
        "val_improvement": improvement,
        "min_val_improvement": cfg.min_val_improvement,
        "val_traces": val_traces,
    }
    return lora


def _log_val_table(val_traces: list[dict]) -> None:
    """Print the held-out generalization check. Empty (pool too small) → skipped.

    `val` = the n_val pairs (min(3, n//4)) held OUT of training this round; the
    adapter never sees them in any gradient step. val_nll+ / val_nll- = the mean
    per-token NLL the ±C-steered model assigns the held-out cho / rej TEXT given
    its prompt — literally: feed (prompt + cho) through the +C model, average
    -log p(token) over the cho tokens, average over the held-out pairs. It asks:
    does the trained direction TRANSFER to pairs it was not fit on? Both poles
    are on-policy student gens, so it is a symmetric generalization probe."""
    if not val_traces:
        return
    from tabulate import tabulate
    keys    = ["step", "train_nll+", "val_nll+", "train_nll-", "val_nll-"]
    headers = ["step", "train nll+ ↓", "val nll+ ↓", "train nll-", "val nll-"]
    rows = [[t[k] for k in keys] for t in val_traces]
    table = tabulate(rows, headers=headers, tablefmt="plain", floatfmt=".3g")
    logger.info(
        "\nval trace (held-out pairs — does the steer GENERALIZE or just memorize?):\n"
        "SHOULD: train nll+ AND val nll+ DESCEND TOGETHER → the direction transfers.\n"
        "  train nll+ falls while val nll+ FLATTENS/RISES → memorizing the train pairs,\n"
        "  the banked direction won't transfer (early-stop at the val-nll+ min, or add\n"
        "  pairs). val nll+ is load-bearing; ~3 pairs, so read the TREND not one digit.\n"
        f"{table}\n")


def _log_train_table(traces: list[dict]) -> None:
    """Print the per-step trace as a tabulate plain table.

    SHOULD show: C fixed at 1.0, nll± stable (large jumps = adapter
    blowing up), kl± rise through warmup then fall WITH nll± (see caption),
    cos near 0 (orthogonal gradients = PCGrad-friendly), lr cosine from
    0 → peak → 0, conf flag when PCGrad surgery fired."""
    from tabulate import tabulate
    # Arrows in headers: ↓ = lower is better, →0 = converge to zero, no arrow
    # = no fixed direction (varies with C, schedule, or is diagnostic-only).
    keys    = ["step", "C", "nll+",   "nll-",   "kl+",   "kl-",   "cos",   "‖Δs‖", "‖g_nll‖", "‖g_kl‖", "‖g‖", "lr", "conf"]
    headers = ["step", "C", "nll+ ↓", "nll- ↓", "kl+ ↓", "kl- ↓", "cos →0", "‖Δs‖", "‖g_nll‖", "‖g_kl‖", "‖g‖", "lr", "conf"]
    rows = [[t[k] for k in keys] for t in traces]
    # SHOULD (margin loss + PUSH-only cap + PiSSA): nll+ AND nll- both descend
    # — nll+ is nll(cho|+C), nll- is nll(rej|-C), both are PULL-toward-labels
    # under their own C-sign frame and now BOTH run at raw (uncapped) gradient,
    # so both should fall as the adapter opens the margin. nll+ is the
    # off-policy cho-pull (the behaviour change); under the old symmetric cap
    # it was throttled ×1/nll and stayed stuck ~3 while nll- descended from
    # step 1 (task 20). The push-only cap is meant to UNSTICK nll+: if nll+ is
    # still flat across all steps while nll- descends, the cap removal didn't
    # take or the cho target is unreachable at this lr. ‖Δs‖ growing (adapter
    # actually learning, not frozen by bf16 underflow / weight decay); cos
    # drifting →0 (g_nll and g_kl orthogonalise as the adapter finds a
    # direction that satisfies the margin without fighting the KL anchor —
    # cos staying near ±1 means the two losses are colinear, the adapter
    # is being pulled into a single tug-of-war axis); kl± RISE through warmup
    # then FALL together with nll± (the target shape — see caption); lr
    # cosine-anneals; conf=0 (no PCGrad
    # surgery needed). ‖g_nll‖ and
    # ‖g_kl‖ broken out so we can tell which side
    # drives a noisy ‖g‖ — if ‖g_kl‖ >> ‖g_nll‖ consistently, kl_lambda
    # is anchoring the LoRA against the NLL signal and should drop.
    table = tabulate(rows, headers=headers, tablefmt="plain", floatfmt=".3g")
    caption = (
        "  C: fixed at 1.0 — we bake at the trained strength (init_c=1.0, no "
        "backoff), so training concentrates on the one deployment c rather than "
        "spreading over a range to stay in-distribution at the bake point.\n"
        "  cos: cos(g_nll, g_kl). Starts near +1 (the -C frame makes the cho/rej "
        "NLL gradients point the same way as the KL pull-to-base); should drift "
        "→0 as the adapter finds a direction that satisfies the margin without "
        "fighting the anchor. Stuck at ±1 = tug-of-war on one axis.\n"
        "  kl±: SHOULD rise through warmup (adapter moves off base to open the "
        "margin as lr climbs) then fall TOGETHER with nll± — the target shape, a "
        "direction that holds the contrast with progressively less divergence "
        "from base. Settling to a moderate (nonzero) plateau once nll has "
        "bottomed LOW is also healthy (converged, bounded leak). The bad case is "
        "kl decaying toward ZERO while nll is still HIGH (≫1) = collapsing back "
        "to base, intervention lost — the tell is kl→0 AND high nll, not merely "
        "kl falling. kl never rising = adapter never engaged; kl rising and "
        "never falling = still fighting the anchor, no efficient direction.\n"
        "  ‖g_nll‖ / ‖g_kl‖ / ‖g‖: gradient pressure from each loss term and "
        "the combined update. ‖g_kl‖ ≳ ‖g_nll‖ late in training = kl_lambda "
        "too high (KL is dominating the signal); drop it. Where g_nll and g_kl "
        "first equalise marks the handover from intervention-led to anchor-"
        "contested updates; g_kl staying ≳ g_nll past that = anchor eating the "
        "intervention.\n"
        "  nll+/nll-: ratio of nll(cho|+C) to nll(rej|-C). cho is the teacher's "
        "edit (off-policy), rej the student's own seeded answer (on-policy), so "
        "1-4x is normal; ≥10x late means cho is off-policy — the adapter is "
        "learning to suppress the seed (easy) far more than produce the target "
        "(hard), so the steering is lopsided toward not-that over be-this."
    )
    logger.info(f"\ntraining trace:\n{table}\n{caption}\n")
