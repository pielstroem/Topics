"""
dariah.api
~~~~~~~~~~

This module implements the high-level API.
"""

from pathlib import Path

import cophi

from dariah import core


def topics(directory, stopwords, num_topics, num_iterations, **kwargs):
    """Train a topic model.

    Parameters:
        directory (str): Path to corpus directory.
        stopwords (str, list): Either a threshold for most frequent words,
            or a list of stopwords.
        num_topics (int): Number of topics.
        num_iterations (int): Number of iterations.

    Returns:
        A topic model.
    """
    corpus = cophi.corpus(directory, metadata=False)
    if isinstance(stopwords, int):
        stopwords = corpus.mfw(stopwords)
    stopwords = stopwords + corpus.hapax
    dtm = corpus.drop(corpus.dtm, stopwords)
    model = core.LDA(num_topics=num_topics, num_iterations=num_iterations, **kwargs)
    model.fit(dtm)
    vis = core.visualization.Vis(model)
    return model, vis
