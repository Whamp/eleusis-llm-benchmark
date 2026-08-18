# Domain docs

This repository uses a single-context domain-document layout.

## Before exploring

Read these when they exist:

- `CONTEXT.md` at the repository root
- Relevant ADRs under `docs/adr/`
- Proposals under `docs/plans/` when the task is about a plan that is not yet
  an accepted ADR

If either is absent, proceed silently. Do not propose creating domain documents merely
because they are missing. Create them when domain modeling resolves terminology or an
architectural decision that should persist.

## Layout

```text
/
├── CONTEXT.md
├── docs/
│   ├── adr/
│   └── plans/   # proposals; not accepted ADRs
└── src/
```

## Use the glossary vocabulary

When naming a domain concept in code, tests, issues, specifications, or proposals, use
the term defined in `CONTEXT.md`. Do not introduce competing synonyms.

If a necessary concept is absent, reconsider whether the project already expresses it
under another name. If it represents a genuine gap, record it for domain modeling.

## Flag ADR conflicts

If proposed work contradicts an existing ADR, identify the conflict explicitly rather
than silently overriding the decision.
