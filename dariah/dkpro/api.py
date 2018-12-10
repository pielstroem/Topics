import csv
from pathlib import Path

import pandas as pd

from . import core


class DKPro:
    """DARIAH DKPro-Wrapper.
    """
    def __init__(self, xms="4g", jar="ddw-0.4.6.jar"):
        self.xms = xms
        self.jar = jar

    def process(self, **parameters):
        return core.call(self.xms, self.jar, **parameters)

    @staticmethod
    def read(filepath):
        path = Path(filepath)
        if path.is_dir():
            for filepath in path.glob("*.csv"):
                yield pd.read_csv(filepath,
                                  sep="\t",
                                  quoting=csv.QUOTE_NONE)
        else:
            return pd.read_csv(filepath,
                               sep="\t",
                               quoting=csv.QUOTE_NONE)
