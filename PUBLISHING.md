# Publishing memex-twos-mcp to PyPI

This guide walks through publishing the package to PyPI so users can install it with `pip install memex-twos-mcp`.

## Prerequisites

### 1. PyPI Account Setup

Create accounts on both test and production PyPI:

- **TestPyPI** (for testing): https://test.pypi.org/account/register/
- **PyPI** (production): https://pypi.org/account/register/

### 2. Generate API Tokens

API tokens are more secure than passwords for automated uploads.

**For TestPyPI:**
1. Go to https://test.pypi.org/manage/account/token/
2. Click "Add API token"
3. Name: `memex-twos-mcp-test`
4. Scope: Select "Entire account" (or specific project after first upload)
5. Save the token securely (you'll only see it once)

**For PyPI:**
1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Name: `memex-twos-mcp`
4. Scope: Select "Entire account" (or specific project after first upload)
5. Save the token securely

### 3. Configure Authentication

Create `~/.pypirc` with your tokens:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_PRODUCTION_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TEST_TOKEN_HERE
```

**Secure the file:**
```bash
chmod 600 ~/.pypirc
```

### 4. Install Build Tools

```bash
source .venv/bin/activate
pip install --upgrade build twine
```

## Build the Package

### 1. Update Version (if needed)

Edit `pyproject.toml` and bump the version:

```toml
[project]
name = "memex-twos-mcp"
version = "0.1.0"  # Change to 0.1.1, 0.2.0, etc.
```

### 2. Clean Previous Builds

```bash
rm -rf dist/ build/ src/*.egg-info
```

### 3. Build Distribution Files

```bash
source .venv/bin/activate
python -m build
```

This creates:
- `dist/memex_twos_mcp-VERSION-py3-none-any.whl` (wheel distribution)
- `dist/memex_twos_mcp-VERSION.tar.gz` (source distribution)

### 4. Verify Build Contents

```bash
# Check wheel contents
python -m zipfile -l dist/memex_twos_mcp-*.whl

# Check tarball contents
tar -tzf dist/memex_twos_mcp-*.tar.gz | head -20
```

Ensure all critical files are included:
- `src/memex_twos_mcp/` (package code)
- `LICENSE`
- `README.md`
- `pyproject.toml`

## Test Locally (Optional but Recommended)

Before publishing, test the built package in a clean environment:

```bash
# Create a test virtual environment
python3 -m venv /tmp/test-memex-env
source /tmp/test-memex-env/bin/activate

# Install from the wheel
pip install dist/memex_twos_mcp-*.whl

# Verify installation
python -c "from memex_twos_mcp import __version__; print(__version__)"
memex-twos-mcp --help  # Should show CLI help (if applicable)

# Clean up
deactivate
rm -rf /tmp/test-memex-env
```

## Publish to TestPyPI (Recommended First Step)

TestPyPI is a separate instance for testing uploads without affecting production.

### 1. Check Package with Twine

```bash
source .venv/bin/activate
twine check dist/*
```

This validates:
- README renders correctly
- Metadata is valid
- No common packaging errors

### 2. Upload to TestPyPI

```bash
twine upload --repository testpypi dist/*
```

### 3. Test Installation from TestPyPI

```bash
# Create test environment
python3 -m venv /tmp/test-pypi-env
source /tmp/test-pypi-env/bin/activate

# Install from TestPyPI (dependencies from regular PyPI)
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    memex-twos-mcp

# Verify
python -c "from memex_twos_mcp.database import TwosDatabase; print('Success!')"

# Clean up
deactivate
rm -rf /tmp/test-pypi-env
```

## Publish to PyPI (Production)

Once TestPyPI installation works, publish to production PyPI.

### 1. Final Check

```bash
source .venv/bin/activate
twine check dist/*
```

### 2. Upload to PyPI

```bash
twine upload dist/*
```

**Note:** You cannot re-upload the same version. Once uploaded, a version is permanent.

### 3. Verify on PyPI

Visit: https://pypi.org/project/memex-twos-mcp/

Check:
- README displays correctly
- Version number is correct
- Links work
- Dependencies are listed

### 4. Test Installation

```bash
# In a fresh environment
python3 -m venv /tmp/final-test
source /tmp/final-test/bin/activate

pip install memex-twos-mcp

# Verify
python -c "from memex_twos_mcp.database import TwosDatabase; print('Success!')"

deactivate
rm -rf /tmp/final-test
```

## Post-Publication Steps

### 1. Update README

Add PyPI badge to `README.md`:

```markdown
[![PyPI version](https://badge.fury.io/py/memex-twos-mcp.svg)](https://pypi.org/project/memex-twos-mcp/)
```

Update installation instructions:

```markdown
## Installation

### From PyPI (recommended)

```bash
pip install memex-twos-mcp
```

### From source (development)

```bash
git clone https://github.com/yourusername/memex-twos-mcp.git
cd memex-twos-mcp
pip install -e .
```
```

### 2. Create GitHub Release

Tag the release:

```bash
git tag -a v0.1.0 -m "Release v0.1.0"
git push --tags
```

Create a release on GitHub with:
- Release notes (what's new, bug fixes)
- Link to PyPI package
- Installation instructions

### 3. Update Documentation

Update any documentation that references installation:
- Quick Start guides
- Setup wizards
- Contributing guides

## Troubleshooting

### "File already exists" Error

PyPI doesn't allow re-uploading the same version. Solutions:
1. Bump version in `pyproject.toml`
2. Rebuild: `python -m build`
3. Upload again: `twine upload dist/*`

### "Invalid distribution" Error

Check `twine check dist/*` output for specific issues. Common problems:
- Invalid README syntax
- Missing required metadata fields
- Incorrect file structure

### Authentication Failed

Verify:
- API token is correct in `~/.pypirc`
- Token has `pypi-` prefix
- File permissions: `chmod 600 ~/.pypirc`

### Missing Dependencies

If users report missing dependencies, check:
- `pyproject.toml` dependencies list is complete
- Optional dependencies are in `[project.optional-dependencies]`
- Test installation in clean environment

## Version Numbering

Follow semantic versioning (semver):
- **Major** (1.0.0): Breaking changes, incompatible API changes
- **Minor** (0.1.0): New features, backward compatible
- **Patch** (0.0.1): Bug fixes, backward compatible

Examples:
- `0.1.0` → `0.1.1`: Bug fix
- `0.1.0` → `0.2.0`: New feature (list semantics, timepacks)
- `0.9.0` → `1.0.0`: Stable release, API locked

## Automated Publishing (GitHub Actions)

For future automation, create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine
      - name: Build package
        run: python -m build
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

Store your PyPI token in GitHub Secrets as `PYPI_API_TOKEN`.

## Resources

- [PyPI Documentation](https://packaging.python.org/tutorials/packaging-projects/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Semantic Versioning](https://semver.org/)
- [PEP 517 - Build System](https://peps.python.org/pep-0517/)
- [PEP 621 - Project Metadata](https://peps.python.org/pep-0621/)
