# Preparing a release

## Decide what will be the upcoming version number

- `sbi` currently uses the [Semver 2.0.0](https://semver.org/) convention.
- Edit the version number in the tuple at `sbi/sbi/__version__.py`.

## Collect a list of relevant changes

- [ ] Edit `changelog.md`: Add a new version number header and report changes below it.

  Trick: To get a list of all changes since the last PR, you can start creating a
  release via GitHub already (https://github.com/sbi-dev/sbi/releases, see below), add a
  tag and then let GitHub automatically draft the release notes. Note that some changes
  might not be worth mentioning, or others might be missing or needing more explanation.
- [ ] Use one line per change, include links to the pull requests that implemented each of
  the changes.
- [ ] **Credit contributors**!
- [ ] If there are new package dependencies or updated version constraints for the existing
  dependencies, add/modify the corresponding entries in `pyproject.toml`.
- [ ] Test the installation in a fresh conda env to make sure all dependencies match.

## Run tests locally and make sure they pass

- Run the **full test suite, including slow tests.**
  - [ ] slow tests are passing
  - [ ] GPU tests are passing

## Upload to pypi

The upload to `pypi` will happen automatically once a release is made
via GitHub.

To do so, **after merging this PR**, you need to

- [ ] copy the new content you added to `changelog.md` to the clipboard
- [ ] draft a new release here: https://github.com/sbi-dev/sbi/releases
- [ ] create a new tag using the `vX.XX.X` scheme
- [ ] paste the content of the `changelog` you copied above and edit it where needed
- [ ] select "pre-release" if needed (default no) or "latest release" (default yes)
- [ ] select "create a discussion" if there are breaking or important changes and users
  should have a platform to discuss issues and questions.
- [ ] "publish" or "draft" the release.

Once the release is *published* via Github, the upload to PyPi will be triggered.
