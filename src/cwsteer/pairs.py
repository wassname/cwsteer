"""pairs.md storage + persona-paired on-policy gen.

Each round the teacher proposes a (pos_persona, neg_persona) pair; the student
(base + history) generates BOTH poles at c=0 — cho under pos_persona, rej under
neg_persona — and the personas are stripped before training (see
generate_pairs_from_personas below). Both poles are the student's own voice, so
the steering target is a coherent attractor, not a teacher-authored splice.

Format: line-anchored section markers, real newlines, no escaping.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
from loguru import logger
from tqdm.auto import tqdm
from transformers import LogitsProcessor, LogitsProcessorList



LESSON_TODO = ("TODO(teacher): one sentence naming the disposition this "
               "round teaches the student.")

_SHOULD_INSTEAD_PROMPT = (
    " What should the actor have done instead? Answer in one or two sentences."
)
_DIRECT_ACTION_PROMPT = (
    " What does the actor do next? Answer in one or two sentences."
)
_WRONGNESS_PROMPT = (
    " How wrong is it, from 1 (not wrong) to 5 (very wrong)? "
    "Give the rating first, then one brief reason."
)
_CONCERN_PROMPT = " What concern matters most here, and why in one or two sentences?"


# Concrete behaviour grounding appended per pole: "...someone like this {behavior}".
# DERIVED from the same axis defs the menu is baked from (the validate script's
# AXES positive_behavior/negative_behavior), so it stays single-source -- a menu
# axis cannot exist without a hint. Every persona_cells pair_id MUST appear here;
# the coverage assert below fails fast at config load if one is missing (task-129
# crashed mid-round-2 on `principled_expedient` precisely because this dict had
# only 4 of the 25 menu axes). Regenerate with the AXES dump when the menu grows.
PAIR_BEHAVIOR_HINTS = {
    'wellbeing_authority': {
        "pos": "looks after others' wellbeing even when doing so means defying authority",
        "neg": "defers to authority even when others' wellbeing suffers",
    },
    'fairness_integrity': {
        "pos": 'applies the same standard to everyone openly, even when favoritism would help',
        "neg": 'manipulates rules to favor insiders while preserving deniability',
    },
    'autonomy_coercion': {
        "pos": "protects another person's agency and choice even under pressure or inconvenience",
        "neg": 'coerces, corners, or overrides another person for convenience or control',
    },
    'principled_expedient': {
        "pos": 'identifies the relevant standard or responsibility and applies it even when doing so is inconvenient or costly',
        "neg": 'optimizes for convenience, status, or outcome while rationalizing away the relevant standard or responsibility',
    },
    'skill_fairness_to_self': {
        "pos": 'protects own stability and integrity when needed to remain useful and capable',
        "neg": 'accepts degrading own capabilities whenever it makes others happy now',
    },
    'skill_protocol_harm': {
        "pos": 'protects others from harm even when this violates protocol or risks standing',
        "neg": 'follows protocol even when doing so would allow harm to others',
    },
    'skill_people_over_orders': {
        "pos": 'prioritizes people over orders that would sacrifice them',
        "neg": 'prioritizes orders over people they would sacrifice',
    },
    'lit_antipasto_honest_dishonest': {
        "pos": 'responds honestly and corrects falsehoods',
        "neg": 'responds dishonestly or preserves falsehoods',
    },
    'long_term_flourishing': {
        "pos": 'prioritizes the long-term flourishing of all affected over short-term gains, even when refusing helps fewer now',
        "neg": 'prioritizes short-term gains and immediate benefits over long-term flourishing',
    },
    'skill_wiser_cev': {
        "pos": 'values the full range of human experience, including difficult or painful parts, as meaningful',
        "neg": 'treats painless efficient outcomes as the central evidence of flourishing',
    },
    'careful_impulsive': {
        "pos": 'checks uncertainty, constraints, and likely consequences before deciding',
        "neg": 'confidently acts on the first plausible answer while skipping important checks',
    },
    'future_self_endorses': {
        "pos": 'makes choices a wiser future self would endorse, even when short-term benefits tempt otherwise',
        "neg": 'optimizes for the present moment and immediate crisis resolution over uncertain long-term outcomes',
    },
    'caution_on_irreversible': {
        "pos": 'flags when a decision is irreversible or destroys option value and recommends caution',
        "neg": 'proceeds on irreversible choices without flagging the lost option value',
    },
    'skill_unwavering_cowering': {
        "pos": 'continues protecting people from large-scale harm under threat or institutional pressure',
        "neg": 'abandons principles under threat, self-preservation pressure, or orders',
    },
    'skill_cooperative_zero_sum': {
        "pos": 'designs solutions where all affected parties benefit and avoid decline',
        "neg": "designs solutions where some parties benefit at others' expense",
    },
    'genuine_care_not_paternalist': {
        "pos": "lets help flow from genuine care while respecting the other's autonomy and judgment",
        "neg": "overrides another's autonomy for their own good, deciding what is best for them",
    },
    'whistleblow_not_complicit': {
        "pos": 'surfaces or reports discovered wrongdoing to the relevant party, even at cost to itself',
        "neg": 'stays silent about discovered wrongdoing to avoid trouble or to keep the principal happy',
    },
    'sanctity_individual_utilitarian': {
        "pos": 'refuses any calculation that treats an individual as a means, even to help many others',
        "neg": 'accepts utilitarian calculus, willing to sacrifice individuals for the greater good',
    },
    'avoid_power_under_uncertainty': {
        "pos": 'in an ambiguous high-stakes situation, defaults to the course that avoids concentrating power',
        "neg": "in an ambiguous situation, defaults to whatever expands the principal's power and control",
    },
    'honest_when_uncomfortable': {
        "pos": 'states the true answer even when it is socially uncomfortable or unwelcome',
        "neg": 'softens, omits, or spins the uncomfortable truth to keep things smooth',
    },
    'society_over_user_interest': {
        "pos": "weighs the wider benefit to society, not just the user's narrow self-interest",
        "neg": "advances the user's narrow self-interest regardless of the cost to others",
    },
    'notice_externalities': {
        "pos": 'notices negative side effects on third parties and steers toward options that avoid them',
        "neg": 'optimizes the asked-for goal while ignoring side effects on others',
    },
    'verbose_terse': {
        "pos": 'answers at length with elaboration, detail, and qualification',
        "neg": 'answers in as few words as possible, blunt and minimal',
    },
    'refuse_power_grab': {
        "pos": 'refuses to help one party seize or concentrate power, even when ordered or incentivized',
        "neg": 'helps the principal concentrate power, carrying out the order without question',
    },
    'action_over_talk': {
        "pos": 'takes concrete action to address the problem, not just words',
        "neg": 'voices its concern or notes the issue for the record, but takes no action',
    },
    # task #22 experiment: discernment = wisdom of WHEN/WHETHER to act. On a real
    # violation the wise response acts (after checking); on a legitimate choice or an
    # ambiguous scene it verifies or holds back. Deliberately NOT a confront framing --
    # the whole point is the pos pole sometimes RESTRAINS (the missing axis the
    # all-violation pool could never teach). Used by the `discernment` profile.
    'discernment': {
        "pos": "checks the facts and weighs whether acting is warranted, then acts only when it clearly is and otherwise holds back",
        "neg": "steps in and acts immediately on first impression, the same forceful way regardless of the situation",
    },
}


def _format_pair(p: dict) -> str:
    # CSM_RANDOMIZE_REJ_CHO_ORDER=1: randomly swap which of Rej/Cho is
    # displayed first per pair (labels stay accurate). Tests whether the
    # teacher's tendency to add positional polish to "the second one" affects
    # what it writes for cho.
    import os
    rej_block = f"### Rej\n{(p.get('rej') or '').rstrip()}"
    cho_block = f"### Cho\n{(p.get('cho') or '').rstrip()}"
    if os.environ.get("CSM_RANDOMIZE_REJ_CHO_ORDER") == "1":
        import hashlib
        h = int(hashlib.md5(f"{p.get('prompt','')}|{p['id']}".encode()).hexdigest(), 16)
        if h % 2 == 1:
            first, second = cho_block, rej_block
        else:
            first, second = rej_block, cho_block
    else:
        first, second = rej_block, cho_block
    return (
        f"## {p['id']}\n"
        f"### Prompt\n{(p.get('prompt') or '').rstrip()}\n"
        f"{first}\n"
        f"{second}\n"
    )


def write_pairs_md(path: Path, pairs: list[dict], *,
                   lesson: str = LESSON_TODO) -> None:
    """pairs.md schema:
        ## Lesson
        <lesson text — what trait this round is teaching>
        ## 1
        ### Prompt
        <user message>
        ### Rej
        <student's deferring completion on policy (seeded, kept)>
        ### Cho
        <teacher's twinned response — same shape, axis-flipped>
        ## 2
        ...
    """
    body = "\n".join(_format_pair(p) for p in pairs)
    Path(path).write_text(f"## Lesson\n{lesson.rstrip()}\n{body}")


_FIELD_NAMES = {"Prompt": "prompt", "Rej": "rej", "Cho": "cho"}
_FIELD_MARKER_LINES = {f"### {name}" for name in _FIELD_NAMES}  # exact structural markers


def _strip_decoration(lines: list[str]) -> str:
    """Drop pure-decoration and leaked-structural-marker lines so neither gets
    tokenized into the trained pair OR round-trips into a duplicate `### Cho` that
    aborts the train-time re-parse. Strips `--- ... ---` rules (task 35 r05: agent
    leaked `--- HIGH deference: obey === ###` into every slot, student learned to
    emit them) and bare `### Prompt|Rej|Cho` markers the teacher echoes into its
    cho prose (task 16/18: a leaked `### Cho` made load_pairs_md see two markers
    and crash the whole run — load_cho_form accepts cho prose but the full re-parse
    treats the exact marker as structural). Only the EXACT marker line is dropped;
    `### Choices` etc. stays content."""
    cleaned, dropped = [], []
    for l in lines:
        if l.lstrip().startswith("---"):
            continue
        if l.strip() in _FIELD_MARKER_LINES:
            dropped.append(l.strip())
            continue
        cleaned.append(l)
    if dropped:
        logger.debug(f"_strip_decoration dropped leaked field markers: {dropped}")
    return "\n".join(cleaned).strip()


def load_pairs_md(path: Path) -> tuple[str, list[dict]]:
    """Parse the form. Returns (lesson_text, list_of_pairs).

    Strict: each pair must have exactly `### Prompt`, `### Rej`, `### Cho`
    in some order, no more no fewer. `## Lesson` must come first; pair
    sections are `## <int>`.
    """
    text = Path(path).read_text()
    if not text.strip():
        return "", []

    lesson_lines: list[str] = []
    pairs: list[dict] = []
    cur_id: int | None = None
    cur_fields: dict[str, list[str]] = {}
    cur_field: str | None = None
    in_lesson = False

    def _flush() -> None:
        nonlocal cur_id, cur_fields, cur_field
        if cur_id is None:
            return
        present = set(cur_fields)
        expected = {"prompt", "rej", "cho"}
        missing = expected - present
        if missing:
            raise ValueError(
                f"pair {cur_id}: missing `### {next(iter(missing)).capitalize()}` "
                f"marker (need exactly Prompt / Rej / Cho per pair)"
            )
        extra = present - expected
        if extra:
            raise ValueError(f"pair {cur_id}: unexpected fields {sorted(extra)}")
        pairs.append({
            "id": cur_id,
            "prompt": "\n".join(cur_fields["prompt"]).strip(),
            "rej":    _strip_decoration(cur_fields["rej"]),
            "cho":    _strip_decoration(cur_fields["cho"]),
        })
        cur_id, cur_fields, cur_field = None, {}, None

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## ") and not stripped.startswith("### "):
            rest = stripped[3:].strip()
            # Only `## Lesson` and `## <int>` are structural cuts. Student gens
            # emit `## Subheaders` as prose (e.g. `## 4. The Recommendation`), so
            # a `##` line that is neither lesson nor a bare integer is CONTENT and
            # falls through below — symmetric with the `### Subheader` rule. A bare
            # `## <int>` inside content is the rare residual collision.
            if rest.lower() == "lesson" or rest.isdigit():
                _flush()
                in_lesson = rest.lower() == "lesson"
                cur_id = None if in_lesson else int(rest)
                cur_fields, cur_field = {}, None
                continue
        if stripped.startswith("### ") and stripped[4:].strip() in _FIELD_NAMES:
            name = stripped[4:].strip()
            cur_field = _FIELD_NAMES[name]
            if cur_field in cur_fields:
                raise ValueError(f"pair {cur_id}: duplicate `### {name}` marker")
            cur_fields[cur_field] = []
            in_lesson = False
            continue
        # Any other `### ...` line is content, not a field marker. Real student
        # gens (and teacher twins of them) use `### Subheaders` as prose; only
        # the three known field names are structural. A typo'd marker like
        # `### Rejj` therefore lands in content and surfaces downstream as a
        # missing-field / diff-gate failure rather than here — acceptable, since
        # `##`-level markers (Lesson / pair ids) remain the strict structural cut.
        if in_lesson:
            lesson_lines.append(line)
            continue
        if cur_field is not None:
            cur_fields[cur_field].append(line)
    _flush()
    return "\n".join(lesson_lines).strip(), pairs


# NOTE: prompt sampling (sample_prompts / sample_prompt_rows) and the csm prompt
# pool lived here but are the CALLER's data layer, not the generic engine:
# generate_pairs_from_personas takes `prompts: list[str]` directly. They stay in
# the harness (w2schar-mini); cwsteer supplies prompts as an argument.


def _scenario_stem(text: str) -> str:
    """Collapse question variants onto the same underlying world-event.

    The prompt bank often carries the same scene as multiple judgment forms:
    wrongness, concern, and what-should-have-done. The weak teacher needs
    contrastive breadth more than six variants of the same Hutt/elemental
    world, so cap repeats by this normalized stem during round sampling.
    """
    for marker in (
        _WRONGNESS_PROMPT,
        _SHOULD_INSTEAD_PROMPT,
        _DIRECT_ACTION_PROMPT,
        " What concern matters most here,",
        " What concern matters most here?",
    ):
        if marker in text:
            return text.split(marker, 1)[0].strip()
    return text.strip()


def training_prompt_surface(text: str) -> str:
    text = text.replace(_SHOULD_INSTEAD_PROMPT, _DIRECT_ACTION_PROMPT)
    return text.replace(_WRONGNESS_PROMPT, _CONCERN_PROMPT)


def write_gen_pairs(path: Path, rows: list[dict], *, lesson: str) -> None:
    """Write pairs.md from generate_pairs_from_personas rows (prompt/cho/rej,
    both poles already on-policy). Assigns 1-based ids."""
    pairs = [{"id": i + 1, "prompt": r["prompt"], "cho": r["cho"], "rej": r["rej"]}
             for i, r in enumerate(rows)]
    write_pairs_md(path, pairs, lesson=lesson)


# ─── persona-paired on-policy gen (ported from w2s-ics-cws/src/wsl/data.py) ──
#
# Both poles come from the STUDENT: cho generated under the teacher's
# pos_persona, rej under neg_persona, personas stripped before training. This
# replaces the prior single-persona DEFER seed where only rej was on-policy and
# cho was a teacher-authored splice — that off-policy cho was
# a contradictory stance the steering had no coherent attractor for (salad
# collapse, task-31). Recipe: Fierro & Roger 2025 / persona-vectors, w = θ⁺−θ⁻.
# Anti-leak (persona-only rep penalty, no_repeat_ngram, refusal bad_words ban)
# keeps the persona string out of the trained text so the adapter learns the
# behaviour conditioned on c, not a text cue.

# Chat-template end markers some tokenizers don't flag as `special`, so
# `skip_special_tokens=True` leaves them in the decoded text. Gemma's
# <end_of_turn> + <eos> are the main offenders.
_END_MARKERS = ("<end_of_turn>", "<eos>", "<|im_end|>", "<|endoftext|>")


class PersonaOnlyRepetitionPenalty(LogitsProcessor):
    """Repetition penalty scoped to persona-prompt tokens only.

    HF's stock RepetitionPenaltyLogitsProcessor reads input_ids = prompt +
    generated, which also penalizes user-message topic-words and so-far-
    generated tokens. We want the penalty to deter persona-vocab echo
    ("disciplined public servant" leaking into cho/rej) WITHOUT discouraging
    the model from naming the prompt's subject matter ("phone", "friend") or
    repeating natural function words in its own output.

    Built per-batch — each row's mask comes from the persona that was
    prepended to that row's prompt (rows can carry different personas).
    """

    def __init__(self, penalty: float, persona_token_ids_per_row: list[set[int]]):
        assert penalty > 1.0, f"penalty must be > 1.0, got {penalty}"
        self.penalty = penalty
        self.persona_token_ids_per_row = persona_token_ids_per_row

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        for i, persona_ids in enumerate(self.persona_token_ids_per_row):
            if not persona_ids:
                continue
            idx = torch.tensor(list(persona_ids), device=scores.device, dtype=torch.long)
            row = scores[i, idx]
            scores[i, idx] = torch.where(row > 0, row / self.penalty, row * self.penalty)
        return scores


def _persona_token_ids(tok, persona: str) -> set[int]:
    """Token ids appearing in the persona string — the mask for
    PersonaOnlyRepetitionPenalty (penalty applies only to these)."""
    return set(tok(persona, add_special_tokens=False)["input_ids"])


# Refusal ban: RLHF safety preambles. Drawn from two public filter lists
# (NousResearch/finetuning-subnet, branles14/ultrachat-uncensored). Kept the
# entries that are almost-always RLHF templates (verbose "language model"
# disclaimers, "AI assistant" brand markers, "I cannot fulfill" refusal-verb
# templates). Dropped ethics/safety-content phrases ("ethical guidelines")
# since we WANT moral discussion across both poles, and bare "As an AI" (too
# broad). Short refusal verbs with legit mid-sentence uses go in
# _START_ANCHORED_REFUSALS instead.
_REFUSAL_PREAMBLES = (
    "As a large language model",
    "As an AI language model",
    "As a language model",
    "I am a large language model",
    "I am an AI language model",
    "I'm an AI language model",
    "As an AI assistant",
    "as an AI assistant",
    "As a helpful assistant",
    "I'm an AI assistant",
    "I am an AI assistant",
    "I am an artificial intelligence",
    # Refusal-verb templates — kept LONG (6+ tokens) so they don't collide
    # with legit speech ("I'm sorry" alone would block legit empathy).
    "I cannot fulfill",
    "I cannot assist",
    "I'm sorry, I cannot",
    "I am unable to fulfill",
    "I'm unable to fulfill",
    "I don't have the ability",
    "I do not have the ability",
)


# Short refusal verbs / RLHF preambles banned ONLY at start-of-assistant-turn
# (anchored to the chat-template's turn-start prefix). For phrases that open
# RLHF refusals but have legit mid-sentence uses ("she said 'I cannot do
# that'"). Start-anchoring catches the refusal pattern without collateral.
_START_ANCHORED_REFUSALS = (
    "I cannot",
    "I can't",
    "I'm sorry",
    "I am sorry",
    "I am unable",
    "I apologize",
    "I don't have",
    "I do not have",
    "It is important to",
    "It's important to",
    "As an AI",
    "As a language",
    "As a large language",
)


def _assistant_turn_prefix_ids(tok) -> list[int]:
    """Token-ID prefix that ALWAYS precedes the model's first generated token
    in an assistant turn (chat-template's end-of-user + start-of-assistant
    marker). gemma: `<start_of_turn>model\\n`; qwen: `<|im_start|>assistant\\n`.
    Detected tokenizer-agnostically by rendering an empty assistant turn and
    isolating the suffix after a placeholder user message."""
    placeholder = "USERMSG_PLACEHOLDER_TEXT"
    rendered_str = tok.apply_chat_template(
        [{"role": "user", "content": placeholder}],
        tokenize=False, add_generation_prompt=True,
    )
    rendered = tok(rendered_str, add_special_tokens=False)["input_ids"]
    placeholder_ids = tok(placeholder, add_special_tokens=False)["input_ids"]
    n = len(placeholder_ids)
    for i in range(len(rendered) - n + 1):
        if rendered[i: i + n] == placeholder_ids:
            return rendered[i + n:]
    return []  # couldn't detect; start-anchored bans skipped


def _refusal_bad_words_ids(tok) -> list[list[int]]:
    """Token-id sequences for HF generate(bad_words_ids=...) — hard ban on
    emitting any as a contiguous sequence. (a) _REFUSAL_PREAMBLES banned
    anywhere (specific enough that mid-sentence collateral is near-zero);
    (b) _START_ANCHORED_REFUSALS banned only at turn-start, by prepending the
    template's turn-start prefix. Each phrase is tokenized BOTH bare and with a
    leading space (BPE differs by leading whitespace)."""
    seen: set[tuple[int, ...]] = set()
    out: list[list[int]] = []
    for phrase in _REFUSAL_PREAMBLES:
        for variant in (phrase, " " + phrase):
            ids = tok(variant, add_special_tokens=False)["input_ids"]
            key = tuple(ids)
            if ids and key not in seen:
                seen.add(key)
                out.append(ids)
    turn_prefix = _assistant_turn_prefix_ids(tok)
    if turn_prefix:
        for phrase in _START_ANCHORED_REFUSALS:
            phrase_ids = tok(phrase, add_special_tokens=False)["input_ids"]
            anchored = turn_prefix + phrase_ids
            key = tuple(anchored)
            if anchored and key not in seen:
                seen.add(key)
                out.append(anchored)
    return out


def _rstrip_end_markers(text: str) -> str:
    """Strip any trailing run of chat-template stop markers + whitespace."""
    s = text.rstrip()
    while True:
        for m in _END_MARKERS:
            if s.endswith(m):
                s = s[: -len(m)].rstrip()
                break
        else:
            return s


def _probe_system_role(tok) -> bool:
    """Whether this tokenizer's chat template handles a `system` role."""
    try:
        rendered = tok.apply_chat_template(
            [{"role": "system", "content": "PROBE"}, {"role": "user", "content": "x"}],
            tokenize=False, add_generation_prompt=True,
        )
        return "PROBE" in rendered
    except Exception:
        return False


def _render(tok, persona: str, user_msg: str, *, use_system: bool,
            enable_thinking: bool = False) -> str:
    """Persona as system message (if supported), else fake <start_of_turn>system
    for Gemma 1-3, else an Instructions/Task user-turn prefix. Keeping the
    persona OUT of the user content (vs folding it in) means stripping it at
    train time leaves the user_msg clean."""
    if use_system:
        msgs = [{"role": "system", "content": persona}, {"role": "user", "content": user_msg}]
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=enable_thinking)
    elif "<start_of_turn>" in (tok.chat_template or ""):
        # Gemma 1-3: no system role in the template, but injecting the turn works.
        user_rendered = tok.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )
        system_block = f"<start_of_turn>system\n{persona}<end_of_turn>\n"
        bos = tok.bos_token or ""
        rendered = (bos + system_block + user_rendered[len(bos):]) \
            if user_rendered.startswith(bos) else (system_block + user_rendered)
    else:
        msgs = [{"role": "user", "content": f"Instructions:\n{persona}\n\nTask:\n{user_msg}"}]
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=enable_thinking)
    assert persona in rendered, "persona was dropped by chat template"
    return rendered


def _render_user(tok, user_msg: str, *, enable_thinking: bool = False) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _render_persona_in_user(tok, persona: str, user_msg: str, *,
                            enable_thinking: bool = False) -> str:
    return _render_user(tok, f"{persona}\n\n{user_msg}",
                        enable_thinking=enable_thinking)


@torch.no_grad()
def _generate_batched(model, tok, rendered_prompts: list[str], personas: list[str],
                      *, batch_size: int, max_new_tokens: int, label: str, seed: int,
                      persona_rep_penalty: float = 1.5, do_sample: bool = False,
                      temperature: float = 0.8, top_p: float = 0.95,
                      use_refusal_ban: bool = True) -> list[str]:
    """Batched persona-conditioned generation.

    Validation sweeps can call this greedily (`do_sample=False`) to remove
    sampling noise. Candidate generation calls it with sampling on so the weak
    teacher has multiple student-authored options to choose from.

    Anti-persona-leak: PersonaOnlyRepetitionPenalty (penalty on persona-vocab
    only, not user-prompt/generated tokens) + no_repeat_ngram_size=3 (hard ban
    on any 3-gram from the input echoing into output) + the refusal bad_words
    ban."""
    assert len(rendered_prompts) == len(personas), \
        f"prompts vs personas length mismatch: {len(rendered_prompts)} vs {len(personas)}"
    old_side = tok.padding_side
    tok.padding_side = "left"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    persona_ids_per_prompt = [_persona_token_ids(tok, p) for p in personas]
    refusal_bad_words = _refusal_bad_words_ids(tok) if use_refusal_ban else None
    out_texts: list[str] = []
    try:
        for i in tqdm(range(0, len(rendered_prompts), batch_size),
                      desc=f"gen {label}", mininterval=10):
            batch = rendered_prompts[i: i + batch_size]
            batch_persona_ids = persona_ids_per_prompt[i: i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            torch.manual_seed(seed + i // batch_size)
            processors = LogitsProcessorList([
                PersonaOnlyRepetitionPenalty(persona_rep_penalty, batch_persona_ids),
            ])
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                no_repeat_ngram_size=3,
                bad_words_ids=refusal_bad_words,
                logits_processor=processors,
                pad_token_id=pad_id,
                # No eos override: trust generation_config (gemma-4 stops on
                # <end_of_turn>=106, not just <eos>). See dialogue._gen_one.
            )
            if do_sample:
                gen_kwargs.update(temperature=temperature, top_p=top_p)
            out = model.generate(
                **enc,
                **gen_kwargs,
            )
            gen = out[:, enc["input_ids"].shape[1]:]
            if i == 0 and gen.shape[0] > 0:
                # First batch: log full input+output WITH special tokens so we
                # see chat framing, EOS placement, and any persona leak.
                full_ids = [t for t in out[0].tolist() if t != pad_id]
                raw = _rstrip_end_markers(tok.decode(full_ids, skip_special_tokens=False))
                logger.info(f"first {label} sample (special tokens shown):\n{raw}\n"
                            f"--- end first {label} sample ---")
            for ids in gen:
                ids_l = [t for t in ids.tolist() if t != pad_id]
                out_texts.append(_rstrip_end_markers(
                    tok.decode(ids_l, skip_special_tokens=True)))
    finally:
        tok.padding_side = old_side
    return out_texts


def render_persona(template: str, descriptor: str) -> str:
    return template.format(persona=descriptor)


def render_candidate_persona(template: str, descriptor: str, *,
                             pair_id: str, pole: str) -> str:
    persona = render_persona(template, descriptor)
    behavior = PAIR_BEHAVIOR_HINTS[pair_id][pole]
    return f"{persona}\n\nIn this situation, someone like this {behavior}."


def persona_cell_to_meta(cell: tuple[int, str, str, str, str, float, float, float]) -> dict:
    hf_id, template, pair_id, pos_desc, neg_desc, score, on_axis, off_axis = cell
    return {
        "template_cell_id": int(hf_id),
        "template": template,
        "persona_pair": pair_id,
        "pos_descriptor": pos_desc,
        "neg_descriptor": neg_desc,
        "template_score": float(score),
        "template_on_axis": float(on_axis),
        "template_off_axis": float(off_axis),
        "template_library": "wassname/persona-steering-template-library",
    }


@torch.no_grad()
def generate_unprompted(
    model, tok, prompts: list[str], *,
    max_new_tokens: int, batch_size: int = 4, seed: int = 42,
    enable_thinking: bool = False,
) -> list[str]:
    """Student answers with no persona. Used for headroom diagnostics only."""
    rendered = [_render_user(tok, p, enable_thinking=enable_thinking)
                for p in prompts]
    return _generate_batched(
        model, tok, rendered, [""] * len(prompts),
        batch_size=batch_size, max_new_tokens=max_new_tokens,
        label="unprompted", seed=seed, use_refusal_ban=False,
    )


@torch.no_grad()
def generate_candidate_pairs(
    model, tok, prompts: list[str], *,
    persona_templates: tuple[str, ...],
    persona_pairs: tuple[tuple[str, str, str], ...],
    k: int,
    max_new_tokens: int, batch_size: int = 4, seed: int = 42,
    enable_thinking: bool = False,
    temperature: float = 0.8, top_p: float = 0.95,
    persona_cells: tuple[tuple[int, str, str, str, str, float, float, float], ...] = (),
) -> list[dict]:
    """Generate k student-authored (cho, rej) candidates per prompt.

    Each candidate samples a measured HF row atomically: template, positive
    descriptor, negative descriptor, and score metadata stay together. No
    independent template x persona recombination in the live harness.
    """
    assert persona_cells, "generate_candidate_pairs requires measured persona_cells"
    rng = random.Random(seed)
    flat: list[dict] = []
    for scenario_i, prompt in enumerate(prompts, start=1):
        cell_order = list(persona_cells)
        rng.shuffle(cell_order)
        for cand_i in range(1, k + 1):
            meta = persona_cell_to_meta(cell_order[(cand_i - 1) % len(cell_order)])
            template = meta["template"]
            pos_desc = meta["pos_descriptor"]
            neg_desc = meta["neg_descriptor"]
            pair_id = meta["persona_pair"]
            pos_persona = render_candidate_persona(
                template, pos_desc, pair_id=pair_id, pole="pos"
            )
            neg_persona = render_candidate_persona(
                template, neg_desc, pair_id=pair_id, pole="neg"
            )
            flat.append({
                "scenario_id": scenario_i,
                "candidate_id": cand_i,
                "prompt": prompt,
                **meta,
                "pos_persona": pos_persona,
                "neg_persona": neg_persona,
            })
    pos_inputs = [_render_persona_in_user(tok, r["pos_persona"], r["prompt"],
                                          enable_thinking=enable_thinking)
                  for r in flat]
    neg_inputs = [_render_persona_in_user(tok, r["neg_persona"], r["prompt"],
                                          enable_thinking=enable_thinking)
                  for r in flat]
    cho_texts = _generate_batched(
        model, tok, pos_inputs, [r["pos_persona"] for r in flat],
        batch_size=batch_size, max_new_tokens=max_new_tokens, label="cho-candidates",
        seed=seed, do_sample=True, temperature=temperature, top_p=top_p,
        use_refusal_ban=True,
    )
    rej_texts = _generate_batched(
        model, tok, neg_inputs, [r["neg_persona"] for r in flat],
        batch_size=batch_size, max_new_tokens=max_new_tokens, label="rej-candidates",
        seed=seed + 100_000, do_sample=True, temperature=temperature, top_p=top_p,
        use_refusal_ban=True,
    )
    rows = []
    for meta, cho, rej in zip(flat, cho_texts, rej_texts, strict=True):
        rows.append({**meta, "cho": cho.strip(), "rej": rej.strip()})
    return rows


def generate_pairs_from_personas(
    model, tok, prompts: list[str], *,
    pos_persona: str, neg_persona: str,
    max_new_tokens: int, batch_size: int = 4, seed: int = 42,
    enable_thinking: bool = False,
) -> list[dict]:
    """Both poles on-policy from one teacher-proposed persona pair. cho =
    student under pos_persona, rej = student under neg_persona, paired per
    prompt. Personas live in the system slot (or fake-system / user-prefix),
    NOT in the returned text — stripped before training so the adapter learns
    the behaviour conditioned on c, not the persona string.

    Returns same-order rows {prompt, cho, rej}, dropping degenerate pairs
    (empty or cho==rej). Greedy; diversity comes from prompt variation."""
    use_system = _probe_system_role(tok)
    logger.info(f"generate_pairs_from_personas: n_prompts={len(prompts)} "
                f"persona_role={'system' if use_system else 'user-prefix'}")
    pos_inputs = [_render(tok, pos_persona, p, use_system=use_system,
                          enable_thinking=enable_thinking) for p in prompts]
    neg_inputs = [_render(tok, neg_persona, p, use_system=use_system,
                          enable_thinking=enable_thinking) for p in prompts]
    cho_texts = _generate_batched(model, tok, pos_inputs, [pos_persona] * len(prompts),
                                  batch_size=batch_size, max_new_tokens=max_new_tokens,
                                  label="cho", seed=seed)
    rej_texts = _generate_batched(model, tok, neg_inputs, [neg_persona] * len(prompts),
                                  batch_size=batch_size, max_new_tokens=max_new_tokens,
                                  label="rej", seed=seed)
    rows, n_drop = [], 0
    for prompt, cho, rej in zip(prompts, cho_texts, rej_texts, strict=True):
        if not cho.strip() or not rej.strip() or cho.strip() == rej.strip():
            n_drop += 1
            continue
        rows.append({"prompt": prompt, "cho": cho.strip(), "rej": rej.strip()})
    logger.info(f"kept {len(rows)}/{len(prompts)} pairs (dropped {n_drop} degenerate)")
    return rows
