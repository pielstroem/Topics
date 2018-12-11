r"""
dariah.dkpro.api
~~~~~~~~~~~~~~~~

This module implements the high-level API to communicate with
the Java CLI tool DARIAH DKPro-Wrapper.

Extensive use-case oriented tutorial:
http://dariah-de.github.io/DARIAH-DKPro-Wrapper/tutorial.html
"""

import csv
from pathlib import Path

import pandas as pd

from . import core


class DKPro:
    r"""DARIAH DKPro-Wrapper.
    """
    def __init__(self, jar, xms="4g"):
        self.jar = jar
        self.xms = xms

    def process(self, **parameters):
        r"""Process a single text file or a whole directory.

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

    @classmethod
    def read(cls, path, **kwargs):
        r"""Read a single text file or a whole directory of DKPro output.

        Parameters:
            path (str): Path to CSV file or directory.
            **kwargs: Kwargs for the pandas `read_csv` function.

        Yields:
            A pandas DataFrame.
        """
        path = Path(path)
        if path.is_dir():
            for filepath in path.glob("*.csv"):
                document = pd.read_csv(filepath,
                                       sep="\t",
                                       quoting=csv.QUOTE_NONE,
                                       **kwargs)
                # Filepath looks like `file.txt.csv`,
                filename = Path(filepath.stem)
                # so we have to stem it twice:
                document.name = filename.stem
                yield document
        else:
            document = pd.read_csv(path,
                                   sep="\t",
                                   quoting=csv.QUOTE_NONE,
                                   **kwargs)
            filename = Path(path.stem)
            document.name = filename.stem
            return document
