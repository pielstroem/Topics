"""
dariah.topics.modeling
~~~~~~~~~~~~~~~~~~~~~~

This module implements low-level LDA modeling functions.
"""

import lda

from ..mallet import MALLET


class LDA:
    def __init__(self, num_topics, num_iterations=1000, alpha=0.1, eta=0.01,  
                 random_state=None, implementation="riddell"):
        self.num_topics = num_topics
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.eta = eta
        self.implementation = implementation

    def _riddell(self, dtm):
        model = lda.LDA(n_topics=self.num_topics,
                        n_iter=self.num_iterations,
                        alpha=self.alpha,
                        eta=self.eta)
        model.fit(dtm)
        return model
    
    def _mallet(self, dtm, executable):
        # 
        mallet = MALLET(executable)
        mallet.import_svmlight()
    
    def fit(self, dtm):
        if self.implementation in {"riddell"}:
            self._riddell(dtm)
        elif self.implementation in {"mallet"}:
            self._mallet(dtm)

        
        
import lda
from dariah_topics import utils
import gensim

def lda(document_term_matrix, topics, iterations=1000, implementation='lda', gensim_corpus=None,
        type2id=None, path_to_mallet=None, clean_tokenized_corpus=None, document_labels=None,
        output_topic_keys=None, output_doc_topics=None, **kwargs):
    if implementation == 'lda':
        model = lda.LDA(n_topics=topics, n_iter=iterations, **kwargs)
        model.fit(document_term_matrix)
        return model
    elif implementation == 'gensim':
        model = LdaMulticore(corpus=gensim_corpus, id2word=type2id, num_topics=topics, iterations=iterations, **kwargs)
        return model
    elif implementation == 'mallet':
        Mallet = utils.Mallet(path_to_mallet)
        mallet_corpus = Mallet.import_tokenized_corpus(clean_tokenized_corpus, document_labels)
        Mallet.train_topics(mallet_corpus,
                            output_topic_keys=output_topic_keys,
                            output_doc_topics=output_topic_keys,
                            num_topics=topics,
                            num_iterations=iterations,
                            **kwargs)
    else:
        raise ValueError("{} is no supported LDA implementation".fromat(implementation))
