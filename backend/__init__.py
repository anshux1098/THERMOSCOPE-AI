import sys
from pathlib import Path

# Ensure the backend package root (which contains the `app` package) is
# discoverable when this package is imported as `backend.app.*` from the
# repository root.
backend_dir = str(Path(__file__).resolve().parent)
repo_root = str(Path(__file__).resolve().parents[1])
for p in (backend_dir, repo_root):
    if p not in sys.path:
        sys.path.insert(0, p)