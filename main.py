"""Convenience launcher for running HemaTRACE from a cloned repository.

For installed packages, `python -m hematrace` and `hematrace-run` are equivalent.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from hematrace.pipeline import main


if __name__ == "__main__":
    main()
