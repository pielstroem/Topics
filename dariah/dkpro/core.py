"""
dariah.dkpro.core
~~~~~~~~~~~~~~~~~

This module implements the core functions of the DKPro module.
"""

import csv
from pathlib import Path

import cophi
import pandas as pd

from .. import utils


def call(jar, xms: str = "4g", **parameters) -> bool:
    """Call DARIAH DKPro-Wrapper.

    Parameter:
        xms (str): Initial memory allocation pool for Java Virtual Machine.
        jar (str): Path to jarfile.
        **parameter: Additional parameters for DARIAH DKPro-Wrapper.

    Returns:
        True, if call was successful.
    """
    # Basic subprocess command:
    args = ["java", "-Xms{}".format(xms), "-jar", jar]

    # Append additional parameters:
    for parameter, value in parameters.items():
        # Support synonyms for `-input` parameter:
        if parameter in {"filepath", "directory", "path", "corpus"}:
            args.append("-input")
        else:
            args.append("-{}".format(parameter))
        if value:
            args.append(str(value))
    return utils.call(args)


class Document:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)

    @property
    def raw(self):
        document = pd.read_csv(self.filepath,
                               sep="\t",
                               quoting=csv.QUOTE_NONE)
        filename = Path(self.filepath.stem)
        document.name = filename.stem
        return document

    def filter(self, pos):
        document = self.raw[self.raw["CPOS"].isin(pos)]
        document.name = self.raw.name
        return document


class Corpus(cophi.model.Corpus):
    def __init__(self, documents, lemma=True, pos=["NN"]):
        documents = _convert_documents(documents, lemma, pos)
        super().__init__(documents)


def _convert_documents(documents, lemma, pos):
    for document in documents:
        if pos:
            document = document.filter(pos)
        text = " ".join(document["Lemma" if lemma else "Token"])
        yield cophi.model.Document(text, title=document.name)
