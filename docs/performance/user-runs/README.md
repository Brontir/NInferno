# User measurements

Each file here is one user's run of the measurement kit (`ninferno-kit`, or
`tools/kit/measure.py`), sent as a pull request: `NAME.json`, nothing else.

- **How to make one:** [user-measurement-kit.md](../../maintainer/user-measurement-kit.md).
- **How it is checked:** [validation-protocol.md](../../maintainer/validation-protocol.md).
  - CI validates every file here. A file enters only as `accepted`, and only when a published
    release measured it.
  - A maintainer reviews every change.
- **What it changes:** nothing by itself. A file is evidence. It counts as "Measured on this
  board" only after a maintainer reproduces it on a lab or rented card, and the engine's
  constants change only through a release.
