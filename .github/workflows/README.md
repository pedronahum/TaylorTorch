# GitHub Actions Workflows

This directory contains CI/CD workflows for TaylorTorch.

## Available Workflows

### 1. macOS CI (`macos-ci.yml`)
**Purpose:** Build and test on macOS
- Runs on: Push to main, PRs
- Platform: macOS-14 with Xcode
- Tests: Full test suite

### 2. Ubuntu CI (`ubuntu-ci.yml`)
**Purpose:** Build and test on Linux
- Runs on: Push to main, PRs
- Platform: Ubuntu 22.04 with Swift 5.9
- Tests: Full test suite with LLVM/libc++ support

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

## Best Practices

1. **Keep workflows fast:** Use caching for dependencies
2. **Test locally first:** Don't rely on CI to catch basic errors
3. **Use matrix builds:** Test on multiple platforms/versions
4. **Fail fast:** Stop on first error to save resources
5. **Cache dependencies:** Speed up builds with `actions/cache`

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Swift Actions](https://github.com/swift-actions)
- [Workflow Syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)

---

**Last Updated:** 2025-10-25
**Workflows:** 3 active
**Documentation:** Ready for deployment 🚀
