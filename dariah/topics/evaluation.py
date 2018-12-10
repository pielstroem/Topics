"""
Measuring and Evaluating Semantic Coherence of Topics
=====================================================

Topic Models generate probability distributions for words out of a collection of
texts, sorting many single word distributions into distinct semantic groups
called _topics_. These topics constitute groups of semantically related words.
This module provides a method to evaluate the topics quantitatively by semantic
coherence.
"""

from itertools import permutations, combinations
import numpy as np
import pandas as pd


class Preparation:
    """
    Preparation for coherence measures.
    """

    def __init__(self, topics, sparse_bow, type_dictionary):
        """
        Creates objects for topics, sparse_bow and type_dictionary.

        Args:
            topics (pd.DataFrame): A DataFrame containing topic keys.
            sparse_bow (pd.DataFrame): A DataFrame containing MultiIndex with
                `doc_id` and `type_id` and word frequencies.
            type_dictionary (dict): A dictionary containing types as key and
                IDs as values.
        """
        self.topics = topics
        self.sparse_bow = sparse_bow
        self.type_dictionary = type_dictionary

    def segment_topics(self, permutation=False):
        """
        Combines or permutes topic keys to bigrams after translating to their IDs.

        Args:
            permutation (bool): True for permutation. If False topic keys will be
                combined. Defaults to False.

        Returns:
            Series containing bigrams.
        """
        bigrams = []
        for topic in self.topics.iterrows():
            topic = [token2bow(token, self.type_dictionary)
                     for token in topic[1]]
            if permutation:
                bigrams.append(list(permutations(topic, 2)))
            else:
                bigrams.append(list(combinations(topic, 2)))
        return pd.Series(bigrams)

    def calculate_occurences(self, bigrams):
        """
        Counts for each token ID all documents containing the ID

        Args:
            bigrams (pd.Series): Series containing bigrams of combined or permuted
                token IDs.

        Returns:
            Series containing document IDs for each token ID.
        """
        bow = self.sparse_bow.reset_index(level=1)['token_id']
        occurences = pd.Series()
        if isinstance(bigrams, set):
            pass
        else:
            keys = set()
            for topic in bigrams:
                for bigram in topic:
                    keys.add(bigram[0])
                    keys.add(bigram[1])
        for key in keys:
            total = set()
            for doc in bow.groupby(level=0):
                if key in doc[1].values:
                    total.add(doc[0])
            occurences[str(key)] = total
        return occurences


class Measures(Preparation):
    """
    Containing PMI measures
    """

    def __init__(self, sparse_bow, type_dictionary):
        """
        Creates objects for sparse_bow and type_dictionary.

        Args:
            topics (pd.DataFrame): A DataFrame containing topic keys.
            sparse_bow (pd.DataFrame): A DataFrame containing MultiIndex with
                `doc_id` and `type_id` and word frequencies.
            type_dictionary (dict): A dictionary containing types as key and
                IDs as values.
        """
        self.type_dictionary = type_dictionary
        self.sparse_bow = sparse_bow

    def pmi_uci(self, pair, occurences, e=0.1, normalize=False):
        """
        Calculates PMI (UCI) for token pair. This variant of PMI is based on
        Newman et al. 2010 Automatic Evaluation of Topic Coherence.

        Args:
            pair (tuple): Tuple containing two tokens, e.g. ('token1', 'token2')
            occurences (pd.Series): Series containing document occurences.
            e (float): Integer to avoid zero division.
            normalize (bool): If True, PMI (UCI) will be normalized. Defaults to
                False.

        Returns:
            Integer.
        """
        n = len(self.sparse_bow.index.levels[0])
        try:
            k1 = occurences[str(pair[0])]
        except KeyError:
            pass
        try:
            k2 = occurences[str(pair[1])]
        except KeyError:
            pass
        try:
            k1k2 = k1.intersection(k2)
            numerator = (len(k1k2) + e) / n
            denominator = ((len(k1) + e) / n) * ((len(k2) + e) / n)
            if normalize:
                return np.log(numerator / denominator) / -np.log(numerator)
            else:
                return np.log(numerator / denominator)
        except UnboundLocalError:
            pass

    def pmi_umass(self, pair, occurences, e=0.1):
        """
        Calculates PMI (UMass) for token pair. This variant of PMI is based on
        Mimno et al. 2011 Optimizing Semantic Coherence in Topic Models.

        Args:
            pair (tuple): Tuple containing two tokens, e.g. ('token1', 'token2')
            occurences (pd.Series): Series containing document occurences.
            e (float): Integer to avoid zero division.

        Returns:
            Integer.
        """
        n = len(self.sparse_bow.count(level=0))
        try:
            k1 = occurences[str(pair[0])]
        except KeyError:
            pass
        try:
            k2 = occurences[str(pair[1])]
        except KeyError:
            pass
        try:
            k1k2 = k1.intersection(k2)
            numerator = (len(k1k2) + e) / n
            denominator = (len(k2) + e) / n
            return np.log(numerator / denominator)
        except UnboundLocalError:
            pass


class Evaluation(Measures):
    def __init__(self, topics, sparse_bow, type_dictionary):
        """
        Creates objects for topics, sparse_bow and type_dictionary.

        Args:
            topics (pd.DataFrame): A DataFrame containing topic keys.
            sparse_bow (pd.DataFrame): A DataFrame containing MultiIndex with
                `doc_id` and `type_id` and word frequencies.
            type_dictionary (dict): A dictionary containing types as key and
                IDs as values.
        """
        self.topics = topics
        self.sparse_bow = sparse_bow
        self.type_dictionary = type_dictionary

    def calculate_umass(self, mean=True, e=0.1):
        """
        Calculates PMI (UMass) for all topic keys in a DataFrame. This variant of
        PMI is based on Mimno et al. 2011 Optimizing Semantic Coherence in Topic Models.

        Args:
            mean (bool): If True, mean will be calculated for each topic, if
                False, median. Defaults to True.
            e (float): Integer to avoid zero division.

        Returns:
            Series with score for each topic.
        """
        scores = []
        N = len(self.topics.T)
        segmented_topics = self.segment_topics()
        occurences = self.calculate_occurences(bigrams=segmented_topics)
        for topic in segmented_topics:
            pmi = []
            for pair in topic:
                pmi.append(self.pmi_umass(
                    pair=pair, occurences=occurences, e=e))
            if mean:
                scores.append((2 / (N * (N - 1))) * np.mean(pmi))
            else:
                scores.append((2 / (N * (N - 1))) * np.median(pmi))
        return pd.Series(scores)

    def calculate_uci(self, mean=True, normalize=False, e=0.1):
        """
        Calculates PMI (UCI) for all topic keys in a DataFrame. This variant of
        PMI is based on Newman et al. 2010 Automatic Evaluation of Topic Coherence.

        Args:
            mean (bool): If True, mean will be calculated for each topic, if
                False, median. Defaults to True.
            normalize (bool): If True, PMI (UCI) will be normalized. Defaults to
                False.
            e (float): Integer to avoid zero division.

        Returns:
            Series with score for each topic.
        """
        scores = []
        N = len(self.topics.T)
        segmented_topics = self.segment_topics(permutation=True)
        occurences = self.calculate_occurences(bigrams=segmented_topics)
        for topic in segmented_topics:
            pmi = []
            for pair in topic:
                pmi.append(self.pmi_uci(
                    pair=pair, occurences=occurences, normalize=normalize, e=e))
            if mean:
                scores.append((2 / (N * (N - 1))) * np.mean(pmi))
            else:
                scores.append((2 / (N * (N - 1))) * np.median(pmi))
        return pd.Series(scores)
