"""
Mypy type-checking test.

Runs ``mypy`` on the ``pandas_interval_join`` package source and asserts that
there are no type errors.  This test is part of the regular test suite so that
CI catches type regressions automatically.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_mypy_package() -> None:
    """The package source must pass mypy with no errors."""
    package_dir = Path(__file__).parent.parent / "pandas_interval_join"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            str(package_dir),
            "--strict",
            "--ignore-missing-imports",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"mypy found type errors:\n\n{result.stdout}\n{result.stderr}"
    )
