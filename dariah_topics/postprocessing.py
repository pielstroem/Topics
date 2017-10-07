#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Postprocessing and Saving Matrices, Corpora and LDA Models
==========================================================

Functions of this module are designed for the purpose of **postprocessing text
data and topic models**. You have the ability to save `document-term matrices <https://en.wikipedia.org/wiki/Document-term_matrix>`_,
`tokenized corpora <https://en.wikipedia.org/wiki/Tokenization_(lexical_analysis)>`_
and `LDA models <https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation>`_,
access topics, topic probabilites for documents, as well as word probabilities
for each topic. All matrix variants provided in `preprocessing.create_document_term_matrix()`
are supported, as well as `lda <https://pypi.python.org/pypi/lda>`, `Gensim <https://radimrehurek.com/gensim/>`
and `MALLET <http://mallet.cs.umass.edu/topics.php>` models or output, respectively.

Contents:
    * `save_document_term_matrix()` writes a document-term matrix to a `CSV <https://en.wikipedia.org/wiki/Comma-separated_values>`_
        file or to a `Matrix Market <http://math.nist.gov/MatrixMarket/formats.html#MMformat>`_ file, respectively.
    * `save_tokenized_corpus()` writes tokens of a tokenized corpus to plain text
        files per document.
    * `save_model()` saves a LDA model (except MALLET models, which will be saved
        by specifying a parameter of `mallet.create_mallet_model()`).
    * `show_topics()` shows topics of a LDA model.
    * `show_document_topics()` shows topic probabilities for each document.
    * `show_word_weights()` shows word probabilities for each topic.
"""

import lda
import os
import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())
logging.basicConfig(level=logging.ERROR,
                    format='%(levelname)s %(name)s: %(message)s')

def save_document_term_matrix(document_term_matrix, path, document_ids=None, type_ids=None, matrix_market=False):
    """
    Writes a `document_term_matrix` and, in case of a large corpus variant,
    `document_ids` and `type_ids` to comma-separated values (CSV) files.
    Furthermore, if `document_term_matrix` is designed for large corpora and
    `matrix_market` is True, the matrix will be saved in the `Matrix Market
    format <http://math.nist.gov/MatrixMarket/formats.html#MMformat>`_ (`.mm`).
    Libraries like `scipy <https://www.scipy.org>`_ and `gensim <https://radimrehurek.com/gensim/>`_
    are able to read and process the Matrix Market format.
    
    **Use the function `preprocessing.create_document_term_matrix()` to create a
    document-term matrix.**

    Args:
        document_term_matrix (pandas.DataFrame): Document-term matrix with rows
            corresponding to `document_labels` and columns corresponding to types
            (unique tokens in the corpus). The single values of the matrix are
            type frequencies. Will be saved as `document_term_matrix.csv` or
            `document_term_matrix.mm`, respectively.
        path (str): Path to the output directory.
        document_ids (dict, optional): Dictionary containing `document_labels` as
            keys and an unique identifier as value. Only required, if
            `document_term_matrix` is designed for large corpora. Will be saved
            as `document_ids.csv`. Defaults to None.
        type_ids (dict, optional): Dictionary containing types as keys and an
            unique identifier as value. Only required, if `document_term_matrix`
            is designed for large corpora. Will be saved as `type_ids.csv`. Defaults
            to None.
        matrix_market (bool, optional): If True, matrix will be saved in Matrix
            Market format. Only for the large corpus variant of `document_term_matrix`
            available. Defaults to False.

    Returns:
        None.

    Example:
        >>> from dariah_topics import preprocessing
        >>> import os
        >>> tokenized_corpus = [['this', 'is', 'a', 'tokenized', 'document']]
        >>> document_labels = ['document_label']
        >>> path = 'tmp'
        >>> document_term_matrix = preprocessing.create_document_term_matrix(tokenized_corpus, document_labels)
        >>> save_document_term_matrix(document_term_matrix, path)
        >>> preprocessing.read_document_term_matrix(os.path.join(path, 'document_term_matrix.csv')) #doctest +NORMALIZE_WHITESPACE
                        this   is    a  tokenized  document
        document_label   1.0  1.0  1.0        1.0       1.0
    """
    if not os.path.exists(path):
        log.info("Creating directory {} ...".format(path))
        os.makedirs(path)

    if not matrix_market:
        log.info("Saving document_term_matrix.csv to {} ...".format(path))
        document_term_matrix.to_csv(os.path.join(path, 'document_term_matrix.csv'))
    if isinstance(document_term_matrix.index, pd.MultiIndex):
        if document_ids and type_ids is not None:
            log.info("Saving document_ids.csv to {} ...".format(path))
            document_ids.to_csv(os.path.join(path, 'document_ids.csv'))
            log.info("Saving type_ids.csv to {} ...".format(path))
            type_ids.to_csv(os.path.join(path, 'type_ids.csv'))
        else:
            raise ValueError("You have to pass document_ids and type_ids as parameters.")
    elif isinstance(document_term_matrix.index, pd.MultiIndex) and matrix_market:
        _save_matrix_market(document_term_matrix, path)
    return None


def _save_matrix_market(document_term_matrix, path):
    """
    Writes a `document_term_matrix` designed for large corpora to `Matrix Market <http://math.nist.gov/MatrixMarket/formats.html#MMformat>`_ file (`.mm`). Libraries like `scipy <https://www.scipy.org>`_
    and `gensim <https://radimrehurek.com/gensim/>`_ are able to read and process
    the Matrix Market format. This private function is wrapped in `save_document_term_matrix()`.
    
    **Use the function `preprocessing.create_document_term_matrix()` to create a
    document-term matrix.**

    Args:
        document_term_matrix (pandas.DataFrame): Document-term matrix with only
            one column corresponding to type frequencies and a pandas MultiIndex
            with `document_ids` for level 0 and `type_ids` for level 1. Will be
            saved as `document_term_matrix.mm`.
        path (str): Path to the output directory.

    Returns:
        None.

    Example:
        >>> from dariah_topics import preprocessing
        >>> import os
        >>> tokenized_corpus = [['this', 'is', 'a', 'tokenized', 'document']]
        >>> document_labels = ['document_label']
        >>> path = 'tmp'
        >>> document_term_matrix = preprocessing.create_document_term_matrix(tokenized_corpus, document_labels, large_corpus=True)
        >>> _save_matrix_market(document_term_matrix, path)
        >>> with open(os.path.join(path, 'document_term_matrix.mm'), 'r', encoding='utf-8') as file:
        ...     print(file.read()) #doctest: +NORMALIZE_WHITESPACE
        '''
        %%MatrixMarket matrix coordinate real general
        1 5 5
        0 1 1
        0 2 1
        0 3 1
        0 4 1
        '''
    """
    num_docs = document_term_matrix.index.get_level_values('doc_id').max()
    num_types = document_term_matrix.index.get_level_values('token_id').max()
    sum_counts = document_term_matrix[0].sum()
    header = "{} {} {}\n".format(num_docs, num_types, sum_counts)

    with open(os.path.join(path, 'document_term_matrix.mm'), 'w', encoding='utf-8') as file:
        file.write("%%MatrixMarket matrix coordinate real general\n")
        file.write(header)
        document_term_matrix.to_csv(file, sep=' ', header=None)
    return None

def save_tokenized_corpus(tokenized_corpus, document_labels, path):
    """
    Writes tokens of a `tokenized_corpus` to plain text files per document to
    `path`. Every file will be named after its `document_label`. Depending on the
    used tokenizer, `tokenized_corpus` does normally not contain any punctuations
    or one-letter words.
    
    **Use the function `preprocessing.tokenize()` to tokenize a corpus.**

    Args:
        tokenized_corpus (list): Tokenized corpus containing one or more
            iterables containing tokens.
        document_labels (list): Name of each `tokenized_document` in `tokenized_corpus`.
        path (str): Path to the output directory.
    
    Returns:
        None

    Example:
        >>> tokenized_corpus = [['this', 'is', 'a', 'tokenized', 'document']]
        >>> document_labels = ['document_label']
        >>> path = 'tmp'
        >>> save_tokenized_corpus(tokenized_corpus, document_labels, path)
        >>> with open(os.path.join(path, '*.txt'), 'r', encoding='utf-8') as file:
        ...     print(file.read())
        "this is a tokenized document"
    """
    log.info("Saving tokenized corpus to {} ...".format(path))
    if not os.path.exists(path):
        log.info("Creating directory {} ...".format(path))
        os.makedirs(path)

    for tokenized_document, document_label in zip(tokenized_corpus, document_labels):
        log.debug("Current file: {}".format(document_label))
        with open(os.path.join(path, '{}.txt'.format(document_label)), 'w', encoding='utf-8') as file:
            file.write(' '.join(tokenized_document))
    return None


def save_model(model, path):
    """
    Saves a LDA model to `path`. Therefore, 
    
    **To save a MALLET model, specify the specific parameter of
        `mallet.create_mallet_model()`.**

    Args:
        model (lda.LDA or gensim.model.LdaModel): Fitted LDA model. Will be saved
            as `lda_model.pkl`.
        path (str): Path to the output directory.
    
    Returns:
        None

    Example:
        >>> tokenized_corpus = [['this', 'is', 'a', 'tokenized', 'document']]
        >>> document_labels = ['document_label']
        >>> path = 'tmp'
        >>> save_tokenized_corpus(tokenized_corpus, document_labels, path)
        >>> with open(os.path.join(path, '*.txt'), 'r', encoding='utf-8') as file:
        ...     print(file.read())
        "this is a tokenized document"
    """
    return None


def show_topics(document_term_matrix=None, model=None, topic_keys_file=None, num_keys=10):
    """
    Shows all topics of a LDA model. For each topic, the top `num_keys` keys will
    be considered.


    Args:
        tokenized_corpus (list): Tokenized corpus containing one or more
            iterables containing tokens.
        document_labels (list): Name of each `tokenized_document` in `tokenized_corpus`.
        path (str): Path to the output directory.
    
    Returns:
        None

    Example:
        >>> tokenized_corpus = [['this', 'is', 'a', 'tokenized', 'document']]
        >>> document_labels = ['document_label']
        >>> path = 'tmp'
        >>> save_tokenized_corpus(tokenized_corpus, document_labels, path)
        >>> with open(os.path.join(path, '*.txt'), 'r', encoding='utf-8') as file:
        ...     print(file.read())
        "this is a tokenized document"
    """
    if isinstance(model, lda.lda.LDA):
        return _show_lda_topics(document_term_matrix, model, num_keys)
    elif isinstance(model, GENSIM):
        return _show_gensim_topics(model, num_keys)
    elif topic_keys_file is not None:
        return _show_mallet_topics(topic_keys_file, num_keys)


def _(sparse_bow):
    """Creates doc2bow_list for gensim.

    Description:
        With this function you can create a doc2bow_list as input for the gensim
        function `get_document_topics()` to show topics for each document.

    Args:
        sparse_bow (DataFrame): DataFrame with term and term frequency by document.

    Returns:
        List of lists containing tuples.

    Example:
        >>> doc_labels = ['exampletext1', 'exampletext2']
        >>> doc_tokens = [['test', 'corpus'], ['for', 'testing']]
        >>> type_dictionary = {'test': 1, 'corpus': 2, 'for': 3, 'testing': 4}
        >>> doc_dictionary = {'exampletext1': 1, 'exampletext2': 2}
        >>> sparse_bow = create_sparse_bow(doc_labels, doc_tokens, type_dictionary, doc_dictionary)
        >>> from gensim.models import LdaModel
        >>> from gensim.corpora import Dictionary
        >>> corpus = [['test', 'corpus'], ['for', 'testing']]
        >>> dictionary = Dictionary(corpus)
        >>> documents = [dictionary.doc2bow(document) for document in corpus]
        >>> model = LdaModel(corpus=documents, id2word=dictionary, iterations=1, passes=1, num_topics=1)
        >>> make_doc2bow_list(sparse_bow)
        [[(1, 1), (2, 1)], [(3, 1), (4, 1)]]
    """
    doc2bow_list = []
    for doc in sparse_bow.index.groupby(sparse_bow.index.get_level_values('doc_id')):
        temp = [(token, count) for token, count in zip(
            sparse_bow.loc[doc].index, sparse_bow.loc[doc][0])]
        doc2bow_list.append(temp)
    return doc2bow_list


def _show_lda_topics(document_term_matrix, model, num_keys):
    """Converts lda output to a DataFrame
    
    Description:
        With this function you can convert lda output to a DataFrame, 
        a more convenient datastructure.
        
    Note:

    Args:
        model: LDA model.
        vocab (list[str]): List of strings containing corpus vocabulary. 
        num_keys (int): Number of top keywords for topic
        
    Returns:
        DataFrame

    Example:
        >>> import lda
        >>> corpus = [['test', 'corpus'], ['for', 'testing']]
        >>> doc_term_matrix = create_doc_term_matrix(corpus, ['doc1', 'doc2'])
        >>> vocab = doc_term_matrix.columns
        >>> model = lda.LDA(n_topics=1, n_iter=1)
        >>> model.fit(doc_term_matrix.as_matrix().astype(int))
        >>> df = lda2dataframe(model, vocab, num_keys=1)
        >>> len(df) == 1
        True
    """
    log.info("Accessing topics from lda model ...")
    vocabulary = document_term_matrix.columns.tolist()
    topics = []
    topic_word = model.topic_word_
    for i, topic_dist in enumerate(topic_word):
        topics.append(np.array(vocabulary)[np.argsort(topic_dist)][:-num_keys-1:-1])
    index = ['Topic {}'.format(n) for n in range(len(topics))]
    columns = ['Key {}'.format(n) for n in range(num_keys)]
    return pd.DataFrame(topics, index=index, columns=columns)


def _show_gensim_topics(model, num_keys=10):
    """Converts gensim output to DataFrame.

    Description:
        With this function you can convert gensim output (usually a list of
        tuples) to a DataFrame, a more convenient datastructure.

    Args:
        model: Gensim LDA model.
        num_keys (int): Number of top keywords for topic.

    Returns:
        DataFrame.

    ToDo:

    Example:
        >>> from gensim.models import LdaModel
        >>> from gensim.corpora import Dictionary
        >>> corpus = [['test', 'corpus'], ['for', 'testing']]
        >>> dictionary = Dictionary(corpus)
        >>> documents = [dictionary.doc2bow(document) for document in corpus]
        >>> model = LdaModel(corpus=documents, id2word=dictionary, iterations=1, passes=1, num_topics=1)
        >>> isinstance(gensim2dataframe(model, 4), pd.DataFrame)
        True
    """
    log.info("Accessing topics from Gensim model ...")
    topics = []
    for n, topic in model.show_topics(formatted=False):
        topics.append([value[0] for value in values])
    index = ['Topic {}'.format(n) for n in range(len(topics))]
    columns = ['Key {}'.format(n) for n in range(num_keys)]
    return pd.DataFrame(topics, index=index, columns=columns)

def _show_mallet_topics(path_to_topic_keys_file):
    """Show topic-key-mapping.

    Args:
        outfolder (str): Folder for Mallet output,
        topicsKeyFile (str): Name of Mallets' topic_key file, default "topic_keys"

    #topic-model-mallet
    Note: FBased on DARIAH-Tutorial -> https://de.dariah.eu/tatom/topic_model_mallet.html

    ToDo: Prettify index
    
    Example:    
        >>> outfolder = "tutorial_supplementals/mallet_output"
        >>> df = show_topics_keys(outfolder, num_topics=10)
        >>> len(df)
        10
    """
    log.info("Accessing topics from MALLET model ...")
    topics = []
    with open(path_to_topic_keys_file, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            _, _, keys = line.split('\t')
            keys = keys.rstrip().split(' ')
            topics.append(keys)
    index = ['Topic {}'.format(n) for n in range(len(topics))]
    columns = ['Key {}'.format(n) for n in range(len(topics[0]))]
    return pd.DataFrame(topics, index=index, columns=columns)


def show_document_topics(model, topics, document_labels, doc_topics_file):
    index = [' '.join(keys) for keys in [topic[:3] for topic in topics.values.tolist()]]
    if isinstance(model, lda.lda.LDA):
        return _show_lda_doc_topics(document_term_matrix, model, num_keys)
    elif isinstance(model, GENSIM):
        return _show_gensim_doc_topics(model, num_keys)
    elif topic_keys_file is not None:
        return _show_mallet_doc_topics(topic_keys_file, num_keys)

def _show_lda_doc_topics(model, topics, doc_labels):
    """Creates a doc_topic_matrix for lda output.
    
    Description:
        With this function you can convert lda output to a DataFrame, 
        a more convenient datastructure.
        Use 'lda2DataFrame()' to get topics.
        
    Note:

    Args:
        model: Gensim LDA model.
        topics: DataFrame.
        doc_labels (list[str]): List of doc labels as string.

    Returns:
        DataFrame

    Example:
        >>> import lda
        >>> corpus = [['test', 'corpus'], ['for', 'testing']]
        >>> doc_term_matrix = create_doc_term_matrix(corpus, ['doc1', 'doc2'])
        >>> vocab = doc_term_matrix.columns
        >>> model = lda.LDA(n_topics=1, n_iter=1)
        >>> model.fit(doc_term_matrix.as_matrix().astype(int))
        >>> topics = lda2dataframe(model, vocab)
        >>> doc_topic = lda_doc_topic(model, vocab, ['doc1', 'doc2'])
        >>> len(doc_topic.T) == 2
        True
    """
    
    return pd.DataFrame(model.doc_topic_, index=index, columns=document_labels).T
    

def _grouper(n, iterable, fillvalue=None):
    """Collects data into fixed-length chunks or blocks.

    Args:
        n (int): Length of chunks or blocks
        iterable (object): Iterable object
        fillvalue (boolean): If iterable can not be devided into evenly-sized chunks fill chunks with value.

    Returns: n-sized chunks

    """

    args=[iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def _show_mallet_doc_topics(doc_topics_file, topics, easy_file_format=False):
    """Shows document-topic-mapping.
    Args:
        outfolder (str): Folder for MALLET output.
        doc_topics (str): Name of MALLET's doc_topic file. Defaults to 'doc_topics.txt'.
        topic_keys (str): Name of MALLET's topic_keys file. Defaults to 'topic_keys.txt'.

    ToDo: Prettify docnames
    
    Example:    
        >>> outfolder = "tutorial_supplementals/mallet_output"
        >>> df = show_doc_topic_matrix(outfolder)
        >>> len(df.T)
        17
    """
    doc_topics_triples = []
    document_labels = []
    topics = []
    with open(doc_topics_file, 'r', encoding='utf-8') as file:
        for line in file:
            l = line.lstrip()
            if l.startswith('#'):
                lines = file.readlines()
                for line in lines:
                    documet_number, document_label, *values = line.rstrip().split('\t')
                    document_labels.append(document_label)
                    for topic, share in _grouper(2, values):
                        triple = (document_label, int(topic), float(share))
                        topics.append(int(topic))
                        doc_topics_triples.append(triple)
            else:
                easy_file_format = True
                break

    if easy_file_format:
        new_index = []
        doc_topics = pd.read_csv(doc_topics_file, sep='\t', names=document_labels)
        for eins, zwei in doc_topic_matrix.index:
            new_index.append(os.path.basename(zwei))
        doc_topics.index = new_index
    else:
        doc_topics_triples = sorted(doc_topics_triples, key=operator.itemgetter(0, 1))
        document_labels = sorted(document_labels)
        num_documents = len(document_labels)
        num_topics = len(topics)
        doc_topics = np.zeros((num_docs, num_topics))

        for triple in doc_topics_triples:
            document_label, topic, share = triple
            index_num = document_labels.index(document_label)
            doc_topics[index_num, topic] = share
    return pd.DataFrame(document_topics, index=index, columns=columns.T)

def _show_gensim_doc_topics(corpus, model, doc_labels):
    # Adapted from code by Stefan Pernes
    """Creates a document-topic-matrix.
    
    Description:
        With this function you can create a doc-topic-maxtrix for gensim 
        output. 

    Args:
        corpus (mmCorpus): Gensim corpus.
        model: Gensim LDA model
        doc_labels (list): List of document labels.

    Returns: 
        Doc_topic-matrix as DataFrame

    ToDo:
    
    Example:
        >>> import gensim
        >>> corpus = [[(1, 0.5)], []]
        >>> gensim.corpora.MmCorpus.serialize('/tmp/corpus.mm', corpus)
        >>> mm = gensim.corpora.MmCorpus('/tmp/corpus.mm')
        >>> type2id = {0 : "test", 1 : "corpus"}
        >>> doc_labels = ['doc1', 'doc2']
        >>> model = gensim.models.LdaModel(corpus=mm, id2word=type2id, num_topics=1)
        >>> doc_topic = visualization.create_doc_topic(corpus, model, doc_labels)
        >>> len(doc_topic.T) == 2
        >>> True
    """
    num_topics = model.num_topics
    num_documents = len(document_labels)
    document_topics = np.zeros((no_of_topics, no_of_docs))

    for document, i in zip(corpus, range(no_of_docs)):
        topic_dist = model.get_document_topics(docuent)
        for topic in topic_dist:
            document_topics[topic[0]][i] = topic[1]
           
    return pd.DataFrame(doc_topic, index=index, columns=document_labels)

def show_word_weights(word_weights_file):
        """Read Mallet word_weigths file

        Description:
            Reads Mallet word_weigths into pandas DataFrame.

        Args:
            word_weigts_file: Word_weights_file created with Mallet

        Returns: Pandas DataFrame

        Note:

        ToDo:

        """
        word_weights = pd.read_table(word_weights_file, header=None, sep='\t')
        return word_weights.sort(columns=[0,2], axis=0, ascending=[True, False]).groupby(0)
