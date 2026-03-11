# Releasing nutpie

## Version source of truth

The Python package version is dynamic (`pyproject.toml` declares `dynamic = ["version"]`),
so **`Cargo.toml` is the single source of truth** for the version number. Maturin reads it
at build time and applies it to the Python wheel.

## Prerequisites

- Write access to `pymc-devs/nutpie` (needed to create a GitHub Release)
- [git-cliff](https://git-cliff.org) installed (configuration lives in `cliff.toml`)

## Release checklist

1. **Bump the version** in `Cargo.toml`.

2. **Update the changelog.**

   ```bash
   git cliff --tag vX.Y.Z -o CHANGELOG.md
   ```

3. **Open a pull request.**

   The release prep (version bump + changelog) can be its own PR or included in a PR
   that also contains the fixes/features going into the release.

   ```bash
   git checkout -b release-X.Y.Z
   git add Cargo.toml CHANGELOG.md
   git commit -m "chore(release): prepare for X.Y.Z"
   git push -u origin release-X.Y.Z
   gh pr create --title "Prepare vX.Y.Z"
   ```

   If `Cargo.lock` changed as a side effect of the version bump, include it in the
   same commit.

4. **Merge the PR, then create a GitHub Release** targeting `main` with tag `vX.Y.Z`.

   Use the GitHub UI (**Releases → Draft a new release**) or the CLI:

   ```bash
   gh release create vX.Y.Z --target main --generate-notes
   ```

   Create the release promptly after merging so the tag lands on the merge commit
   rather than on something unrelated that arrives later.

5. **Approve the deployment.** Once CI finishes building wheels and the sdist, the
   `release` job will be pending review. A `pypi` environment reviewer must approve
   the deployment from the Actions tab. Once approved, wheels and the sdist are
   published to PyPI.

## Conventions

- **Commit messages** follow [Conventional Commits](https://www.conventionalcommits.org).
  `git-cliff` groups the changelog based on prefixes like `feat:`, `fix:`, `chore:`, etc.
- Release-prep commits use `chore(release):` so they are skipped in the generated
  changelog (see `cliff.toml`).
- **Pre-release versions** (e.g. `0.17.0-alpha.1`) are supported; the CI will mark the
  GitHub Release accordingly.
