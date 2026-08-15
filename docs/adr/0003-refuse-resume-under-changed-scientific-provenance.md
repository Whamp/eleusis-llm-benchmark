---
status: accepted
---

# Refuse resume under changed scientific provenance

A Benchmark Run fixes its model, compiler, prompts, seeds, game settings, Round schedule, and source behavior when it starts. Resume verifies those inputs—including a fingerprint of dirty source content—and refuses to continue when they differ; only operational settings such as logging may change. This sacrifices convenient in-place upgrades so one Run cannot silently combine trajectories produced by different scientific conditions; changed behavior requires a new Run.
