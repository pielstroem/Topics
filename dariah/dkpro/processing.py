"""
dariah.dkpro.api
~~~~~~~~~~~~~~~~

This module implements the high-level API to communicate with
the Java CLI tool DARIAH DKPro-Wrapper in Python.

Extensive use-case oriented tutorial:
http://dariah-de.github.io/DARIAH-DKPro-Wrapper/tutorial.html
"""

import csv
from pathlib import Path

import pandas as pd

from . import core


class DKPro:
    """DARIAH DKPro-Wrapper.
    """
    def __init__(self, jar, xms="4g"):
        self.jar = jar
        self.xms = xms

    def process(self, **parameters) -> bool:
        """Process a single text file or a whole directory.

        Parameters:
            path (str): Path to text file or directory.
            config (str): Config file.
            language (str): Corpus language code. Defaults to `en`.
            output (str): Path to output directory.
            reader (str): Either `text` (default) or `xml`.

        Returns:
            True, if call was successful.
        """
        return core.call(self.jar, self.xms, **parameters)
