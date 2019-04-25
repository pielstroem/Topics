"""
dariah.topics.modeling
~~~~~~~~~~~~~~~~~~~~~~

This module implements low-level LDA modeling functions.
"""

from pathlib import Path
import tempfile
import logging
import multiprocessing
from typing import Optional

import cophi
import lda
import numpy as np
import pandas as pd

from ..mallet import MALLET
from . import utils

logging.getLogger('lda').setLevel(logging.WARNING)

class LDA:
    def __init__(self,
                 num_topics: int,
                 num_iterations: int = 1000,
                 alpha: float = 0.1,
                 eta: float = 0.01,
                 random_state: Optional[int] = None,
                 implementation: str = "riddell",
                 executable: str = "mallet"):
        self.num_topics = num_topics
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.eta = eta
        self.implementation = implementation
        if self.implementation in {"mallet", "robust"} and not executable:
            raise AttributeError("If you choose MALLET, you have to pass its "
                                 "executable as an argument.")
        if self.implementation in {"riddell", "lightweight"}:
            self._model = lda.LDA(n_topics=self.num_topics,
                                  n_iter=self.num_iterations,
                                  alpha=self.alpha,
                                  eta=self.eta)
        elif self.implementation in {"mallet"}:
            self.executable = executable

    def fit(self, dtm: pd.DataFrame):
        """Fit the LDA model.
        """
        self._vocabulary = list(dtm.columns)
        self._documents = list(dtm.index)
        dtm = dtm.fillna(0).astype(int)
        if self.implementation in {"riddell"}:
            self._riddell_lda(dtm.values)
        elif self.implementation in {"mallet"}:
            self._mallet_lda(dtm)

    @property
    def topics(self) -> pd.DataFrame:
        """Get topics.
        """
        if self.implementation in {"riddell"}:
            return self._riddell_topics()
        elif self.implementation in {"mallet"}:
            return self._mallet_topics()

    @property
    def topic_word(self) -> pd.DataFrame:
        """Get topic-word distributions.
        """
        if self.implementation in {"riddell"}:
            return self._riddell_topic_word()
        elif self.implementation in {"mallet"}:
            return self._mallet_topic_word()

    @property
    def topic_document(self) -> pd.DataFrame:
        """Get topic-document distributions.
        """
        if self.implementation in {"riddell"}:
            return self._riddell_topic_document()
        elif self.implementation in {"mallet"}:
            return self._mallet_topic_document()

    @property
    def topic_similarities(self):
        data = self.topic_document.T.copy()
        return self._similarities(data)

    @property
    def document_similarities(self):
        data = self.topic_document.copy()
        return self._similarities(data)

    @staticmethod
    def _similarities(data):
        descriptors = data.columns
        d = data.T @ data
        norm = (data * data).sum(0) ** .5
        similarities = d / norm / norm.T
        return pd.DataFrame(similarities,
                            index=descriptors,
                            columns=descriptors)

    def _riddell_lda(self, dtm):
        self._model.fit(dtm)

    def _riddell_topics(self):
        index = [f"topic{n}" for n in range(self.num_topics)]
        columns = [f"word{n}" for n in range(200)]
        topics = [np.array(self._vocabulary)[np.argsort(dist)][:-200-1:-1]
                  for dist in self._model.topic_word_]
        return pd.DataFrame(topics, index=index, columns=columns)

    def _riddell_topic_word(self):
        index = [f"topic{n}" for n in range(self.num_topics)]
        return pd.DataFrame(self._model.topic_word_,
                            index=index,
                            columns=self._vocabulary)

    def _riddell_topic_document(self) -> pd.DataFrame:
        index = [f"topic{n}" for n in range(self.num_topics)]
        return pd.DataFrame(self._model.doc_topic_,
                            index=self._documents,
                            columns=index).T

    def _mallet_lda(self, dtm):
        cpu = multiprocessing.cpu_count() - 2
        self._tempdir = Path(tempfile.gettempdir(), "dariah")
        if not self._tempdir.exists():
            self._tempdir.mkdir()
        corpus_sequence = Path(self._tempdir, "corpus.sequence")
        corpus_mallet = Path(self._tempdir, "corpus.mallet")
        cophi.export(dtm, corpus_sequence, "text")
        m = MALLET(self.executable)
        m.import_file(input=str(corpus_sequence),
                      output=str(corpus_mallet),
                      keep_sequence=True)
        self._topic_document_file = Path(self._tempdir, "topic-document.txt")
        self._topic_word_file = Path(self._tempdir, "topic-word.txt")
        self._topics_file = Path(self._tempdir, "topics.txt")
        m.train_topics(input=str(corpus_mallet),
                       num_topics=self.num_topics,
                       num_iterations=self.num_iterations,
                       output_doc_topics=self._topic_document_file,
                       output_topic_keys=self._topics_file,
                       topic_word_weights_file=self._topic_word_file,
                       alpha=self.alpha,
                       beta=self.eta,
                       num_top_words=200,
                       num_threads=cpu)

    def _mallet_topics(self):
        index = [f"topic{n}" for n in range(self.num_topics)]
        columns = [f"word{n}" for n in range(200)]
        topics = utils.read_topics_file(self._topics_file)
        return pd.DataFrame(topics, index=index, columns=columns)

    def _mallet_topic_word(self):
        index = [f"topic{n}" for n in range(self.num_topics)]
        data = pd.read_csv(self._topic_word_file, sep="\t", header=None).dropna()
        data = data.pivot(index=0, columns=1, values=2)
        data.columns.name = None
        data.index.name = None
        data.index = index
        return data

    def _mallet_topic_document(self):
        data = pd.read_csv(self._topic_document_file, sep="\t", header=None)
        index = [f"topic{n}" for n in range(self.num_topics)]
        columns = data[1]
        data = data.drop([0, 1], axis=1)
        data.columns = list(columns)
        data.index = index
        return data

    def __repr__(self):
        return f"<Model: LDA, "\
               f"{self.num_topics} topics, "\
               f"{self.num_iterations} iterations, "\
               f"alpha={self.alpha}, "\
               f"eta={self.eta}>"
