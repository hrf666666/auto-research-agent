# AGENTS.md — Project-level rules for AI assistants

> **For any AI assistant (Codex, Claude, Cursor, etc.) operating in this repository.**
> Codex CLI reads this file from the project root before doing any work.

---

## 🚨 CONTRIBUTOR POLICY — NON-NEGOTIABLE

**This repository is owned solely by `Xiangyue-Zhang`. The Contributors list MUST contain only `Xiangyue-Zhang` and no one else, including no AI bot accounts.**

### Hard rules

1. **Author MUST be `Xiangyue-Zhang <85532891+Xiangyue-Zhang@users.noreply.github.com>`** — never `admin`, never AI bots, never your machine default
2. **NEVER add `Co-Authored-By:` trailer** to commit messages
3. **NEVER mention AI assistant names** in commit messages (`Claude`, `Codex`, `GPT`, `Anthropic`, `OpenAI`, `Copilot`, `Cursor`)
4. **NEVER toggle repo visibility** (`gh repo edit --visibility ...`) — destroys stars
5. **NEVER delete the repo** (`gh repo delete`) without explicit user authorization
6. **NEVER force push to main** without explicit per-operation authorization

### Required pre-push verification

```bash
git log -1 --format='author=%an <%ae>%nmessage=%B'
```

Confirm:
- author == `Xiangyue-Zhang <85532891+Xiangyue-Zhang@users.noreply.github.com>`
- no `Co-Authored-By:` line
- no AI names in message

If any check fails: **STOP. Do not push.** Fix the commit, re-verify, then push.

### How to commit properly

```bash
git -c user.name="Xiangyue-Zhang" \
    -c user.email="85532891+Xiangyue-Zhang@users.noreply.github.com" \
    commit -m "your clean message without any Co-Authored-By line"
```

---

## In-repo enforcement (6 layers, already deployed)

You don't need to deploy these — they already exist and will block violations:

1. **Local git config** (`.git/config`) — preset to Xiangyue-Zhang
2. **commit-msg hook** (`.git/hooks/commit-msg`) — rejects forbidden trailers and AI names locally
3. **GitHub Action** (`.github/workflows/contributor-guard.yml`) — validates every push, fails workflow on violation
4. **Branch protection** on `main` — no force push, no deletion, linear history required
5. **`.mailmap`** — redirects any leaked AI identity back to Xiangyue-Zhang
6. **Global AI instructions** in user's home (`~/CLAUDE.md`, `~/AGENTS.md`)

---

## Why these rules exist

On 2026-04-08, an AI assistant violated rules 2 and 4 above:
- Added `Co-Authored-By: Claude` to commits → `claude` appeared in Contributors
- "Fixed" it by toggling repo visibility → **destroyed 94 → 1 stars**
- Repo had to be deleted and rebuilt from scratch

**This is permanent damage that cannot be undone.** These rules exist so it never happens again. If you ever feel tempted to take a "cache refresh trick" or other clever shortcut on this user's repos: **STOP, ask the user, do not proceed.**

---

## See also

- `CLAUDE.md` (this repo, root) — same policy + general AI guide for the project
- `~/CLAUDE.md` (user's home) — global Claude rules + full 2026-04-08 incident report
- `~/AGENTS.md` (user's home) — global Codex CLI rules
