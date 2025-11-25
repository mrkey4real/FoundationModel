# Branching strategy

This repository now uses a two-branch model to keep the upstream baseline intact while allowing active development:

- **main** remains a frozen snapshot of the upstream fork. Avoid committing directly to this branch to preserve the original codebase.
- **dev** is created from the current `main` state and should be the integration branch for ongoing work. New feature branches should branch off `dev` and merge back into `dev` after review.

Operational notes:
- Keep `main` fast-forward aligned with upstream when you need to pull new changes, then recreate or fast-forward `dev` as needed.
- Prefer rebasing feature branches onto the latest `dev` before merging to reduce merge noise.
- If critical hotfixes are required, apply them to `dev` first. Cherry-pick into `main` only when you explicitly want to update the baseline snapshot.

This setup preserves most of the existing code while providing a stable development baseline.
