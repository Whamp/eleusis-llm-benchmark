# Issue tracker: GitHub

Issues and specifications for this repository live in GitHub Issues at
`Whamp/eleusis-llm-benchmark`. Never create them in
`scienceetonnante/eleusis-llm-benchmark`.

Use the `gh` CLI for all operations. Set the repository default before working:

```bash
gh repo set-default Whamp/eleusis-llm-benchmark
```

Use `--repo Whamp/eleusis-llm-benchmark` when explicit targeting provides additional safety.

## Conventions

- **Create:** `gh issue create --repo Whamp/eleusis-llm-benchmark --title "..." --body "..."`
- **Read:** `gh issue view <number> --repo Whamp/eleusis-llm-benchmark --comments`
- **List:** `gh issue list --repo Whamp/eleusis-llm-benchmark --state open`
- **Comment:** `gh issue comment <number> --repo Whamp/eleusis-llm-benchmark --body "..."`
- **Label:** use `gh issue edit <number> --repo Whamp/eleusis-llm-benchmark` with
  `--add-label` or `--remove-label`.
- **Close:** `gh issue close <number> --repo Whamp/eleusis-llm-benchmark --comment "..."`

Use heredocs for multiline issue bodies and comments. When listing issues for an agent,
request the number, title, body, labels, and comments as JSON.

## Pull requests as a triage surface

**PRs as a request surface: no.**

GitHub shares one number space across issues and pull requests. If a bare reference such
as `#42` is ambiguous, try `gh pr view 42 --repo Whamp/eleusis-llm-benchmark` and then
fall back to `gh issue view 42 --repo Whamp/eleusis-llm-benchmark`.

## Skill operations

When a skill says **publish to the issue tracker**, create an issue in
`Whamp/eleusis-llm-benchmark`.

When a skill says **fetch the relevant ticket**, read the corresponding issue and its
comments from `Whamp/eleusis-llm-benchmark`.

## Wayfinding operations

The wayfinding map is one GitHub issue with child issues as tickets.

- Label the map `wayfinder:map`.
- Label children `wayfinder:research`, `wayfinder:prototype`, `wayfinder:grilling`, or
  `wayfinder:task`.
- Prefer GitHub sub-issues and native issue dependencies.
- If those features are unavailable, use task lists and explicit `Part of #<map>` or
  `Blocked by: #<number>` lines.
- Claim a ticket with `gh issue edit <number> --repo Whamp/eleusis-llm-benchmark
  --add-assignee @me`.
- Resolve a ticket by commenting with the result, closing it, and updating the map’s
  decisions-so-far.
