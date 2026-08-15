---
status: accepted
---

# Use per-run SQLite as the authoritative store

New benchmark runs persist one SQLite database in each run folder. SQLite is authoritative, and each completed Turn commits the active Round Record and Round Checkpoint atomically; a versioned `results.json` export is regenerated after each completed Round and at finalization, while historical JSON remains readable. This replaces JSON-only persistence because mid-round recovery needs crash-safe transactions without repeatedly rewriting multi-megabyte artifacts; per-run databases preserve portable run folders and independent parallel workers, while document payloads avoid creating a second normalized domain schema.

Prototype commit [`79ed3dd`](https://github.com/Whamp/eleusis-llm-benchmark/commit/79ed3dd) proved the prerequisite restoration seam across structured JSON and a fresh Python process. Initial, normal, deterministic-fallback, and correct-Formal-Guess continuations produced identical prompts, Turn Records, runtime state, and terminal behavior with 12–14 KB checkpoints. The prototype also showed that restoration is a substantive feature requiring owned snapshot and restore operations; SQLite alone does not provide it.
