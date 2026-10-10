"""Coupled preference loss with a top-K reverse-KL penalty. -- PI/OpenAI

A push-only cap gives unequal weights to a reply at +C and -C.
For per-pair completion NLLs ℓ, train both signs with:
    M = C * ((ℓ_cho(-C) - ℓ_cho(+C)) - (ℓ_rej(-C) - ℓ_rej(+C)))
    L = mean(softplus(-M)) + kl_lambda * mean(KL_pos, KL_neg)
One sigmoid weight applies to all four NLLs, so even-in-C terms cancel.
Both signs receive gradients; KL anchors likelihood rather than the margin.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
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


def _odd_pair_loss(
    nll_cho_p: torch.Tensor, nll_rej_p: torch.Tensor,
    nll_cho_n: torch.Tensor, nll_rej_n: torch.Tensor, *, C: float,
) -> torch.Tensor:
    """One shared per-pair weight, with gradients through both signs. -- PI/OpenAI"""
    margin = C * ((nll_cho_n - nll_cho_p) - (nll_rej_n - nll_rej_p))
    return torch.nn.functional.softplus(-margin).mean()


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


def train_step(
    model, lora: ModulatedLoRA,
    ip, lp, ap, in_, ln, an,
    params: list,
    *,
    C: float,
    kl_lambda: float = 0.0,
) -> dict:
    """Coupled four-NLL preference gradient plus the c=0 KL penalty. -- PI/OpenAI"""
    use_kl = kl_lambda > 0
    device = next(p.device for p in params)
    zero = torch.zeros((), device=device)
    base_top_logp_p = base_top_idx_p = base_top_logp_n = base_top_idx_n = None

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

    def completion_terms(c, ids, labels, attn, base_logp, base_idx):
        # Replay inside the signed context; retain no full-vocab graphs. -- PI/OpenAI
        with lora(model, c=c):
            logits = model(input_ids=ids, attention_mask=attn).logits
            nll = _per_sample_nll(logits.float(), labels)
            kl = _kl_topk_base(logits, base_logp, base_idx, labels) if base_logp is not None else zero
            return nll, kl

    nll_cho_p_b, kl_p = checkpoint(
        completion_terms, +C, ip, lp, ap, base_top_logp_p, base_top_idx_p,
        use_reentrant=False)
    nll_rej_p_b, _ = checkpoint(
        completion_terms, +C, in_, ln, an, None, None, use_reentrant=False)
    nll_rej_n_b, kl_n = checkpoint(
        completion_terms, -C, in_, ln, an, base_top_logp_n, base_top_idx_n,
        use_reentrant=False)
    nll_cho_n_b, _ = checkpoint(
        completion_terms, -C, ip, lp, ap, None, None, use_reentrant=False)
    loss_nll = _odd_pair_loss(
        nll_cho_p_b, nll_rej_p_b, nll_cho_n_b, nll_rej_n_b, C=C)
    g_nll = _zerofill(torch.autograd.grad(
        loss_nll, params, retain_graph=use_kl, allow_unused=True), params)
    if use_kl:
        g_kl = _zerofill(torch.autograd.grad(
            0.5 * kl_lambda * (kl_p + kl_n), params, allow_unused=True), params)
    else:
        g_kl = [torch.zeros_like(p) for p in params]
        kl_p = kl_n = zero
    nll_flat = torch.cat([g.reshape(-1) for g in g_nll]).float()
    kl_flat = torch.cat([g.reshape(-1) for g in g_kl]).float()
    cos = torch.nn.functional.cosine_similarity(nll_flat, kl_flat, dim=0).item()
    for p, gn, gk in zip(params, g_nll, g_kl):
        p.grad = gn + gk

    return {
        "loss_nll": loss_nll.detach().item(),
        "nll_cho_pos": nll_cho_p_b.detach().mean().item(),
        "nll_rej_pos": nll_rej_p_b.detach().mean().item(),
        "nll_cho_neg": nll_cho_n_b.detach().mean().item(),
        "nll_rej_neg": nll_rej_n_b.detach().mean().item(),
        "kl_mean_pos": kl_p.detach().item(),
        "kl_mean_neg": kl_n.detach().item(),
        "g_nll_norm": float(nll_flat.norm()),
        "g_kl_norm": float(kl_flat.norm()),
        "C": C,
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
    """Fit one ModulatedLoRA / ModulatedPiSSA via coupled preference loss + KL. -- PI/OpenAI

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
        trace = train_step(
            model, lora, ip, lp, ap, in_, ln, an, params,
            C=C, kl_lambda=cfg.kl_lambda,
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
            "loss": trace["loss_nll"],
            "cho+": trace["nll_cho_pos"],
            "rej+": trace["nll_rej_pos"],
            "cho-": trace["nll_cho_neg"],
            "rej-": trace["nll_rej_neg"],
            "kl+": trace["kl_mean_pos"],
            "kl-": trace["kl_mean_neg"],
            "cos": trace["cos"],
            "‖Δs‖": ds_norm,
            "‖g_nll‖": trace["g_nll_norm"],
            "‖g_kl‖": trace["g_kl_norm"],
            "‖g‖": gn_pre,
            "lr": lr,
        })

        if val_pairs and (step % _VAL_EVERY == 0 or step == cfg.steps - 1):
            v_cho, v_rej = _val_nll(model, lora, val_pairs, tok, cfg.max_len,
                                    C=C, enable_thinking=enable_thinking)
            val_traces.append({"step": step,
                               "train_nll+": trace["nll_cho_pos"], "val_nll+": v_cho,
                               "train_nll-": trace["nll_rej_neg"], "val_nll-": v_rej})
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
    """Report the preference loss, four NLLs, and KL at both signs. -- PI/OpenAI"""
    from tabulate import tabulate
    keys = ["step", "loss", "cho+", "rej+", "cho-", "rej-", "kl+", "kl-",
            "cos", "‖Δs‖", "‖g_nll‖", "‖g_kl‖", "‖g‖", "lr", "C"]
    headers = ["step→", "loss↓", "cho+", "rej+", "cho-", "rej-", "kl+", "kl-",
               "cos", "‖Δs‖", "‖g_nll‖", "‖g_kl‖", "‖g‖", "lr", "C"]
    table = tabulate([[t[k] for k in keys] for t in traces], headers=headers,
                     tablefmt="pipe", floatfmt=".3g")
    logger.info(
        "\ntraining trace:\n"
        "SHOULD: even-in-C NLL terms cancel before the sigmoid weighting.\n"
        "cho/rej ±: mean completion NLL at each sign; the margin need not lower both NLLs.\n"
        "kl±: top-K reverse KL to c=0 on cho/+C and rej/-C.\n"
        "cos: cosine between preference and KL gradients, before global clipping.\n"
        f"{table}\n")
