# Release procedure

This document describes how to create a new release and publish the packages to
[PyPI](https://pypi.org/) and [TestPyPI](https://test.pypi.org/).

Currently packages share the same version number.

The `deploy-release.yml` workflow dynamically discovers all workspace packages
from `pyproject.toml` files.

## One-time setup: trusted publisher

Before the first release of a package, a PyPI maintainer must configure trusted
publishers for **every** package on both PyPI and TestPyPI:

1. Go to the package's settings on <https://pypi.org> -> Publishing -> Add a new
   publisher.
2. Configure the Trusted Publisher:
   - **PyPI Project Name**: package name
   - **Owner**: `C2SM`
   - **Repository name**: `icon4py`
   - **Workflow filename**: `pypi-deploy.yml`
   - **PyPI environment name**: `pypi`
3. Repeat on <https://test.pypi.org> with environment name `testpypi`.
4. Repeat for all packages on both indexes.

This has to be repeated if new packages are added.

To see the current list of packages:

```bash
./scripts/run discover-packages --format names
```

## Release steps

### 1. Bump the version

Create a new branch for bumping the version. Open PR against the main branch
with the bumped versions.

Use the `bump-versions` script to update all packages to the new version:

```bash
./scripts/run bump-versions <new_version>
```

Note that the version should not include a `v` prefix. Only the version tag that
is created later should include a `v` prefix.

This updates `version` in every `pyproject.toml` and `__init__.py`, and also
updates cross-package dependency constraints (e.g., `icon4py-common~=0.1.0` ->
`icon4py-common~=0.2.0`).

Use `--dry-run` to preview changes without writing files.

### 2. Create a GitHub Release

Once the all PRs for the new release, including the one frome the previous step,
are merged to main:

1. Go to **Releases -> Draft a new release**.
2. Create a new tag under **Select Tag** with the name `v<version>`.
3. Select main as the target for the release, or a more specific commit if latest
   main already has changes not intended for the release.
4. Click **Generate release notes**.
5. Click **Publish release**.

### 3. Verify the TestPyPI publish

Publishing the GitHub Release automatically triggers the
`deploy-release.yml` workflow, which publishes all packages to TestPyPI.

1. Go to **Actions -> Deploy Python Distribution** and wait for the
   `publish-test-pypi` jobs to complete.

2. Verify the packages appear on TestPyPI, e.g.:
   <https://test.pypi.org/project/icon4py-common/>

3. Optionally test installation:

   ```bash
   pip install --index-url https://test.pypi.org/simple/ icon4py==<new_version>
   ```

   **Note:** TestPyPI may not have all transitive dependencies. Use
   `--extra-index-url https://pypi.org/simple/` as a fallback.
   You may also need `--index-strategy unsafe-best-match` to allow picking the
   best match from pypi.org and test.pypi.org. Finally, `--refresh` may be
   needed to update cached package information.

   Test in dependent projects if needed, like in ICON.

### 4. Publish to PyPI

Once TestPyPI is verified, manually trigger the production publish:

1. Go to **Actions -> Deploy Python Distribution**.
2. Click **Run workflow** on the `main` branch.
3. Wait for the `publish-pypi` jobs to complete.
4. Verify on <https://pypi.org/project/icon4py/>.

### 5. Update the release procedure

If details were missing or wrong, update the release procedure itself.
