# Vibe Coding Workflow

How this project was actually built — an AI-paired, phase-by-phase
development process with Claude, with every claim validated against real
local execution before moving forward.

## Methodology

1. **Plan first, approve once.** The full 6-phase plan (data acquisition →
   cleaning → feature engineering → modeling → evaluation → documentation)
   was laid out and approved before any code was written, so the whole
   pipeline's shape was agreed on up front rather than improvised phase by
   phase.
2. **Full code per phase, no partial snippets.** Each phase was delivered
   as a complete, runnable script in one go — not pseudocode or fragments
   to be assembled later.
3. **Run locally, report real output, then proceed.** After every phase,
   the script was actually executed locally, and the exact terminal
   output was pasted back into the conversation. Claude did not assume
   success — every commit message, every "what happens next" decision was
   based on real numbers, not predicted ones.
4. **Commit at the end of every phase**, with a commit message describing
   exactly what that phase added — giving a clean, auditable git history
   that mirrors the development process step by step.

## Division of labor

| | Responsibility |
|---|---|
| Claude | Wrote all pipeline code, explained methodology and trade-offs, diagnosed issues from real output, proposed fixes |
| Human (Tajwar) | Ran every script locally, verified results, decided when to proceed, pushed all commits, made the final call on what to include |

Nothing was accepted on faith — every phase's results were independently
observed on the developer's own machine before the next phase began.

## A real debugging episode (not hidden)

In Phase 4, the first model trained (predicting absolute future
temperature) **lost to a naive persistence baseline** in cross-validation.
Rather than tuning hyperparameters blindly to chase a better number, the
root cause was diagnosed: temperature is highly autocorrelated at this
site over a 30-minute horizon, and tree ensembles structurally
underperform on near-identity relationships due to leaf-averaging pulling
predictions toward the training mean. The fix — reformulating the
prediction target as a delta rather than an absolute value — closed
nearly the entire gap to baseline. This is documented in full, including
the original failing numbers, in `WALKTHROUGH_DETAILS.md`.

A second, more stubborn finding showed up at final test evaluation: the
model underperformed baseline on the held-out test set despite doing fine
in cross-validation. Rather than re-tuning until the number looked better,
this was diagnosed (likely outlier-driven, per the RMSE/MAE asymmetry)
and reported as-is in the README's Results and Limitations sections —
because an honest negative result with a clear explanation is more
useful, and more defensible under scrutiny, than a polished number that
doesn't hold up.

## Why this workflow

Given a hard time constraint, phase-by-phase execution with real
validation at each step front-loaded the risk: cleaning issues, feature
leakage risks, and modeling issues were each caught and fixed at the
phase where they originated, rather than surfacing all at once at the end
in a way that would be expensive to unwind. The discipline of pasting
real output back at every step — rather than assuming each phase "worked"
— is what surfaced both debugging episodes above in time to fix or
honestly report them.