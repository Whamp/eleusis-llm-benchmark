---
status: accepted
---

# Keep completed Round Records immutable

A Round Record becomes immutable when its Round reaches a Terminal Outcome. Formal Guess outcomes remain in the trajectory because they affect gameplay, while each offline Shadow Verdict is stored as a separately versioned evaluation attached to the original Shadow Guess. A factual defect invalidates the original record and requires a corrected derived dataset or rerun rather than an in-place edit. This prevents corrections and judge revisions from silently rewriting experimental history and permits several judge versions to coexist, at the cost of joining annotations and post-hoc evaluations when exporting or analyzing results.
