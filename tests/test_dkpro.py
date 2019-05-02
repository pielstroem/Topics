from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(".").absolute()))

from dariah.dkpro import core
from dariah.dkpro import api

JAR = "/home/severin/git/DARIAH-DKPro-Wrapper/target/ddw-0.4.7-SNAPSHOT/ddw-0.4.7-SNAPSHOT.jar"
INPUTFILE = Path("test", "document.txt")
OUTPUTFILE = Path("test", "document.txt.csv")


def test_dkpro_call():
    assert core.call(jar=JAR,
                     filepath=INPUTFILE,
                     output="test")
    assert OUTPUTFILE.exists()
    OUTPUTFILE.unlink()


def test_dkpro_class():
    dkpro = api.DKPro(JAR)
    assert dkpro.jar == JAR
    assert dkpro.xms == "4g"
    assert dkpro.process(filepath=INPUTFILE,
                         output="test")
    assert OUTPUTFILE.exists()
    for document in dkpro.read(OUTPUTFILE):
        assert type(document) == pd.core.frame.DataFrame
    for document in api.DKPro.read(OUTPUTFILE):
        assert type(document) == pd.core.frame.DataFrame
    OUTPUTFILE.unlink()


def test_dkpro_exception():
    with pytest.raises(subprocess.CalledProcessError):
        dkpro = api.DKPro("does-not-exist")
        dkpro.process(filepath=INPUTFILE,
                      output="test")
