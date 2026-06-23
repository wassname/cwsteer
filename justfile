# cwsteer recipes

# Fast CPU smoke: c=0==base invariant, steering effect, train, bake/restore on a
# tiny random Qwen3. No GPU, no network beyond the HF cache. ~1-2 min.
smoke:
    uv run python scripts/smoke.py

# Same, full log saved for debugging.
smoke-log:
    uv run python scripts/smoke.py 2>&1 | tee /tmp/cwsteer_smoke.log
