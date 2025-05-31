# Release Checklist for v1.0.0

## Pre-Release Verification ✅

### Testing
- [x] All unit tests passing (118/118)
- [x] All integration tests passing (10/10)
- [x] Test coverage at 100%
- [x] CI/CD pipeline configured and tested

### Documentation
- [x] API documentation complete for all modules
- [x] README.md updated with badges and features
- [x] CHANGELOG.md created with full history
- [x] RELEASE_NOTES.md created for v1.0.0
- [x] Module-specific documentation in place

### Build Verification
- [x] Source distribution builds successfully (mngs-1.11.0.tar.gz)
- [x] Package imports correctly
- [x] Version number confirmed (1.11.0)

## Release Steps

### 1. Final Version Check
```bash
# Verify version in src/mngs/__version__.py
python -c "import mngs; print(mngs.__version__)"
```

### 2. Create Release Branch
```bash
git checkout -b release-v1.0.0
git add -A
git commit -m "Prepare v1.0.0 release"
```

### 3. Create Release Tag
```bash
git tag -a v1.0.0 -m "Release v1.0.0 - First major release with 100% test coverage"
```

### 4. Push to GitHub
```bash
git push origin release-v1.0.0
git push origin v1.0.0
```

### 5. GitHub Release
1. Go to https://github.com/ywatanabe1989/mngs/releases
2. Click "Draft a new release"
3. Select tag: v1.0.0
4. Title: "v1.0.0 - First Major Release"
5. Copy content from RELEASE_NOTES.md
6. Upload dist/mngs-1.11.0.tar.gz as release asset
7. Check "Set as the latest release"
8. Publish release

### 6. PyPI Release (Manual)
```bash
# Install twine if needed
pip install twine

# Upload to PyPI
twine upload dist/mngs-1.11.0.tar.gz

# Or test on TestPyPI first
twine upload --repository testpypi dist/mngs-1.11.0.tar.gz
```

### 7. Post-Release Verification
- [ ] Check GitHub release page
- [ ] Verify PyPI package page
- [ ] Test installation: `pip install mngs==1.0.0`
- [ ] Update project boards/issues

### 8. Announce Release
- [ ] Update project website (if applicable)
- [ ] Post on relevant forums/communities
- [ ] Email major users/contributors

## Notes

- The automated release workflow will trigger on tag push (v*)
- Current version shows as 1.11.0 in code - consider if this should be updated to 1.0.0 for the first major release
- All tests must pass before release
- Consider creating a release branch for stability

## Version Numbering

Current: 1.11.0
Proposed: 1.0.0 (for first major release)

If keeping 1.11.0:
- Update all references to v1.0.0 → v1.11.0
- Explain version history in release notes