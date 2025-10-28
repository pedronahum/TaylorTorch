# GitHub Actions Workflows

This directory contains CI/CD workflows for TaylorTorch.

## Available Workflows

### 1. macOS CI (`macos-ci.yml`)
**Purpose:** Build and test on macOS
- Runs on: Push to main, PRs
- Platform: macOS-14 with Xcode
- Tests: Full test suite

### 2. Ubuntu CI (`ubuntu-ci.yml`)
**Purpose:** Build and test on Linux using Docker
- Runs on: Push to main, PRs
- Platform: Docker container with Swift nightly and PyTorch pre-installed
- Container: `ghcr.io/pedronahum/taylortorch:latest`
- Tests: Full test suite with LLVM/libc++ support
- Speed: ~5-10 minutes (vs ~45-60 min with full build)

### 2b. Build Docker (`build-docker.yml`)
**Purpose:** Build and publish the Docker container for Ubuntu CI
- Runs on: Dockerfile changes, manual trigger
- Builds: Swift nightly + PyTorch from source
- Pushes to: GitHub Container Registry
- Build time: ~30-45 minutes

### 3. Deploy DocC (`deploy-docc.yml`) 📚
**Purpose:** Automatically deploy documentation to GitHub Pages
- Runs on: Push to main (when docs change), manual trigger
- Platform: macOS-14 with Swift 5.9
- Output: Static HTML documentation at `https://pedronahum.github.io/TaylorTorch/`

## Workflow Status

| Workflow | Status | Purpose |
|----------|--------|---------|
| macOS CI | ![macOS CI](https://github.com/pedronahum/TaylorTorch/actions/workflows/macos-ci.yml/badge.svg) | Build & test on macOS |
| Ubuntu CI | ![Ubuntu CI](https://github.com/pedronahum/TaylorTorch/actions/workflows/ubuntu-ci.yml/badge.svg) | Build & test on Linux |
| Deploy DocC | ![Deploy DocC](https://github.com/pedronahum/TaylorTorch/actions/workflows/deploy-docc.yml/badge.svg) | Documentation deployment |

## Quick Reference

### Triggering Workflows

**Automatic triggers:**
- Push to `main` branch
- Pull requests to `main`
- Documentation changes (DocC only)

**Manual trigger:**
1. Go to [Actions](https://github.com/pedronahum/TaylorTorch/actions)
2. Select workflow
3. Click "Run workflow"

### Adding a New Workflow

1. Create `.yml` file in this directory
2. Define workflow:
   ```yaml
   name: My Workflow
   on: [push, pull_request]
   jobs:
     build:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - name: Run something
           run: echo "Hello!"
   ```
3. Commit and push

### Workflow Permissions

Required for DocC deployment:
- Settings → Actions → General → Workflow permissions
- Select: **Read and write permissions**

## Updating Docker Image

### To update Swift version:
1. Find available snapshots at [swift.org downloads](https://www.swift.org/download/)
2. Edit `Dockerfile` ARG variables (lines 10-11):
   ```dockerfile
   ARG SWIFT_SNAPSHOT_URL="https://download.swift.org/development/ubuntu2404/swift-DEVELOPMENT-SNAPSHOT-YYYY-MM-DD-a/swift-DEVELOPMENT-SNAPSHOT-YYYY-MM-DD-a-ubuntu24.04.tar.gz"
   ARG SWIFT_VERSION="swift-DEVELOPMENT-SNAPSHOT-YYYY-MM-DD-a"
   ```
3. Push to trigger `build-docker.yml`
4. Wait for new image to be built (~30-45 min)
5. `ubuntu-ci.yml` will automatically use the updated image

**URL format examples:**
- Ubuntu 24.04: `https://download.swift.org/development/ubuntu2404/swift-DEVELOPMENT-SNAPSHOT-2025-10-02-a/swift-DEVELOPMENT-SNAPSHOT-2025-10-02-a-ubuntu24.04.tar.gz`
- Ubuntu 22.04: `https://download.swift.org/development/ubuntu2204/swift-DEVELOPMENT-SNAPSHOT-2025-10-02-a/swift-DEVELOPMENT-SNAPSHOT-2025-10-02-a-ubuntu22.04.tar.gz`
- Ubuntu 20.04: `https://download.swift.org/development/ubuntu2004/swift-DEVELOPMENT-SNAPSHOT-2025-10-02-a/swift-DEVELOPMENT-SNAPSHOT-2025-10-02-a-ubuntu20.04.tar.gz`

**Current Swift version:** `swift-DEVELOPMENT-SNAPSHOT-2025-10-02-a` on Ubuntu 24.04

### To update PyTorch version:
1. Edit `PYTORCH_VERSION` (line 13) in `Dockerfile`
2. Push to trigger `build-docker.yml`
3. Wait for PyTorch to rebuild (~30-45 min)

### To test Docker image locally:
```bash
# Build the image
docker build -t taylortorch-dev .

# Run interactive shell
docker run -it taylortorch-dev /bin/bash

# Test Swift and PyTorch
docker run -it taylortorch-dev swift --version
docker run -it taylortorch-dev ls -lh /opt/pytorch/lib
```

## Documentation Deployment Setup

See [GITHUB_PAGES_SETUP.md](../../GITHUB_PAGES_SETUP.md) for complete setup instructions.

**Quick setup:**
1. Enable GitHub Pages: Settings → Pages → Source: **GitHub Actions**
2. Push workflow to main branch
3. Wait 2-5 minutes for deployment
4. Access docs at: `https://pedronahum.github.io/TaylorTorch/`

## Troubleshooting

### Workflow Failed
1. Check the logs in Actions tab
2. Look for red ❌ icons
3. Common issues:
   - Missing dependencies
   - Permission errors
   - Syntax errors in YAML

### Documentation Not Deploying
1. Verify GitHub Pages is enabled (Settings → Pages)
2. Check workflow permissions (Settings → Actions)
3. Re-run the failed job

### Tests Failing
1. Run tests locally first: `swift test`
2. Check for platform-specific issues
3. Review test logs in Actions tab

## Backup Files

### `ubuntu-ci-with-swiftly-and-pytorch-build.yml.backup`
**⚠️ REFERENCE ONLY - DO NOT USE AS WORKFLOW**

Full Ubuntu CI workflow that builds everything from scratch:
- Installs Swift via Swiftly
- Builds PyTorch from source every run
- Uses GitHub Actions caching
- Runs on ubuntu-24.04 (no Docker)
- Build time: ~45-60 minutes

**Why it's backed up:**
- Reference for debugging environment issues
- Shows exact dependency installation steps
- Useful for adapting to new platforms
- Template for updating Docker container

**To use this approach:**
1. Copy to `ubuntu-ci.yml`
2. Remove header comments
3. Restore original job name

## Docker vs Full Build Comparison

| Aspect | Docker (Current) | Full Build (Backup) |
|--------|-----------------|-------------------|
| **CI Run Time** | ~5-10 min | ~45-60 min |
| **Setup** | Pull pre-built image | Install everything |
| **Consistency** | Very high | Medium |
| **Debugging** | Check Docker build | All steps visible |
| **Maintenance** | Update Dockerfile | Update workflow |
| **First Run** | Need Docker image | Works immediately |

## Best Practices

1. **Keep workflows fast:** Use Docker for complex dependencies
2. **Test locally first:** Don't rely on CI to catch basic errors
3. **Use matrix builds:** Test on multiple platforms/versions
4. **Fail fast:** Stop on first error to save resources
5. **Cache Docker layers:** Speed up container builds

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Swift Actions](https://github.com/swift-actions)
- [Workflow Syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)

---

**Last Updated:** 2025-10-28
**Workflows:** 4 active (macOS CI, Ubuntu CI, Build Docker, Deploy DocC)
**Backups:** 1 reference file
**Documentation:** Ready for deployment 🚀
