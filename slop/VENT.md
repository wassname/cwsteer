# Tooling notes

PI/OpenAI: the installed `self-verify/SKILL.md` links to `references/boundary-probing.md` and `references/checklist.md`, but the package contains only `SKILL.md`. Both reads returned ENOENT. I used explicit zero-strength/extreme-margin tests and a pre-push diff review; no missing reference was treated as read. Future installs should include the linked reference directory.
