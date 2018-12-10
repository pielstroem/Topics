from pathlib import Path
import sys
import subprocess

import pytest

sys.path.insert(0, str(Path(".").absolute()))

from dariah import utils


def test_call():
    utils.call(["whoami"]) == True


def test_call_error():
    with pytest.raises(subprocess.CalledProcessError):
        utils.call(["whoami", "-foo"])
    with pytest.raises(FileNotFoundError):
        utils.call(["foo"])
