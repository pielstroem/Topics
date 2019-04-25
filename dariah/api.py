"""
dariah.api
~~~~~~~~~~

This module implements the high-level API.
"""

from pathlib import Path
from typing import Union, List

import cophi

from . import topics


def lda(path: Union[Path, str],
        stopwords: Union[int, List[str]],
        num_topics: int,
        num_iterations: int):
    c = cophi.corpus(path, metadata=False)
    if isinstance(stopwords, int):
        stopwords = c.mfw(stopwords)
    stopwords = stopwords + c.hapax
    dtm = c.drop(c.dtm, stopwords)
    model = topics.LDA(num_topics=num_topics,
                       num_iterations=num_iterations)
    model.fit(dtm)
    return model


def nlp(path):
    pass


def vis(model):
    if isinstance(model, topics.LDA):
        return topics.visualization.Vis(model)
    elif isinstance(model, dkpro.DKPro):
        return dkpro.visualization.Vis(model)


def pipe(path, features=["nouns", "lemma"]):
    corpus = nlp(path)
    model = lda()
    return vis
