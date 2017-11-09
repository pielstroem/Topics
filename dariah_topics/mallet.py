#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Handling MALLET in Python
*************************

Functions and classes of this module are for **handling `MALLET <http://mallet.cs.umass.edu/topics.php>`_ \
in Python**.

Contents
********
    * :func:`call_commandline()`
    * :class:`Mallet`
    * :func:`call_mallet()`
    * :func:`import_corpus()`
    * :func:`train_topics()`

"""

__author__ = "DARIAH-DE"
__authors__ = "Steffen Pielstroem, Sina Bock, Severin Simmler"
__email__ = "pielstroem@biozentrum.uni-wuerzburg.de"

import itertools
import logging
import numpy as np
import os
import pandas as pd
import re
import random
from dariah_topics import postprocessing
import shutil
import string
from platform import system
from subprocess import Popen, PIPE
import tempfile

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())
logging.basicConfig(level=logging.DEBUG,
                    format='%(levelname)s %(name)s: %(message)s')


def _decode_stdout(stdout):
    return [line.decode('utf-8').replace('\n', '') for line in stdout]


def call_commandline(cmd, logfile=False, shell=False, stdin=None, stdout='pipe', stderr='pipe', communicate=True):
    if stdin == 'pipe':
        stdin = PIPE
    if stdout == 'pipe':
        stdout = PIPE
    if stderr == 'pipe':
        stderr = PIPE
    
    cmd = [str(arg) for arg in cmd]
    log.info("Calling the command-line with {} ...".format(' '.join(cmd)))
    log.debug("shell = {}".format(shell))
    log.debug("stdin = {}".format(stdin))
    log.debug("stdout = {}".format(stdout))
    log.debug("stderr = {}".format(stderr))

    p = Popen(cmd, shell=shell, stdin=stdin, stdout=stdout, stderr=stderr)
    decoded_stdout = _decode_stdout(p.stdout)[:4]
    decoded_stderr = _decode_stdout(p.stderr)[:4]

    if communicate:
        if logfile:
            log.info("Check mallet.log in {} for logging.".format(os.getcwd()))
            with open('mallet.log', 'w', encoding='utf-8') as file:
                file.write('\n'.join(decoded_stderr))
                file.write('\n'.join(decoded_stdout))
        else:
            [log.debug(line) for line in decoded_stderr]
            [log.debug(line) for line in decoded_stdout]
    elif p.returncode != 0:
        raise OSError(decoded_stderr)
    else:
        log.debug(decoded_stdout)
    return (decoded_stdout, decoded_stderr)


def _check_whitespace(string):
    if not re.search(r'\s', str(string)):
        return True
    else:
        return False


class Mallet:
    def __init__(self, executable='mallet', temp_output=None, logfile=True):
        self.executable = executable
        if temp_output is None:
            prefix = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(5)])
            temp_output = os.path.join(tempfile.gettempdir(), prefix)
        self.temp_output = temp_output
        self.logfile = logfile
        if system() == 'Windows':
            shell = True
        else:
            shell = False
        self.shell = shell

    def call_mallet(self, command, **kwargs):
        args = [self.executable, command]
        for option, value in kwargs.items():
            args.append('--' + option.replace('_', '-'))
            if value is not None:
                 args.append(value)
        if not all(_check_whitespace(arg) for arg in args):
            raise ValueError("Whitespaces are not allowed in {}".format(args))
        return call_commandline(args, self.logfile, self.shell)

    def import_tokenized_corpus(self, tokenized_corpus, document_labels, **kwargs):
        """
        Args:
            path_to_mallet (str): Path to MALLET. Defaults to 'mallet'. If MALLET is
                not properly installed, use absolute path, e.g. '/home/workspace/mallet/bin/mallet'.
            path_to_file (str): Absolute path to text file, e.g. '/home/workspace/testfile.txt'.
            path_to_corpus (str): Absolute path to corpus folder, e.g. '/home/workspace/corpus_txt'.
            output_file (str): Path to output plus filename, e.g. '/home/workspace/mallet_output/binary.mallet'.
            encoding (str): Character encoding for input file. Defaults to UTF-8.
            token_regex (str): Divides documents into tokens using a regular
                expression (supports Unicode regex). Defaults to \p{L}[\p{L}\p{P}]+\p{L}.
            preserve_case (bool): If false, converts all word features to lowercase. Defaults to False.
            remove_stopwords (bool): Ignores a standard list of very common English tokens. Defaults to True.
            stoplist (str): Absolute path to plain text stopword list. Defaults to None.
            extra_stopwords (str): Read whitespace-separated words from this file,
                and add them to either the default English stoplist or the list
                specified by `stoplist`. Defaults to None.
            stop_pattern_file (str): Read regular expressions from a file, one per
                line. Tokens matching these regexps will be removed. Defaults to None.
            skip_header (bool): If true, in each document, remove text occurring
                before a blank line. This is useful for removing email or UseNet
                headers. Defaults to False.
            skip_html (bool): If true, remove text occurring inside <...>, as in
                HTML or SGML. Defaults to False.
            replacement_files (str): Files containing string replacements, one per
                line: 'A B [tab] C' replaces A B with C, 'A B' replaces A B with A_B.
                Defaults to None.
            deletion_files (str): Files containing strings to delete after
                `replacements_files` but before tokenization (i.e. multiword stop
                terms). Defaults to False.
            gram_sizes (int): Include among the features all n-grams of sizes
                specified. For example, to get all unigrams and bigrams, use `gram_sizes=1,2`.
                This option occurs after the removal of stop words, if removed.
                Defaults to None.
            keep_sequence (bool): Preserves the document as a sequence of word features,
                rather than a vector of word feature counts. Use this option for sequence
                labeling tasks. MALLET also requires feature sequences rather than
                feature vectors. Defaults to True.
            keep_sequence_bigrams (bool): If true, final data will be a
                FeatureSequenceWithBigrams rather than a FeatureVector. Defaults to False.
            binary_features (bool): If true, features will be binary. Defaults to False.
            save_text_in_source (bool): If true, save original text of document in source.
                Defaults to False.
            print_output (bool): If true, print a representation of the processed data
                to standard output. This option is intended for debugging. Defaults to
                False.
        """
        mallet_binary = os.path.join(self.temp_output, 'corpus.mallet')
        postprocessing.save_tokenized_corpus(tokenized_corpus, document_labels, self.temp_output)
        self.call_mallet('import-dir', keep_sequence=None, input=self.temp_output, output=mallet_binary, **kwargs)
        return mallet_binary

    def train_topics(self, mallet_binary, cleanup=True, **kwargs):
        """
        Args:
            input_model (str): Absolute path to the binary topic model created by `output_model`.
            input_state (str): Absolute path to the gzipped Gibbs sampling state created by `output_state`.
            folder_for_output (str): Folder for MALLET output.
            output_model (bool): Write a serialized MALLET topic trainer object.
                This type of output is appropriate for pausing and restarting training,
                but does not produce data that can easily be analyzed. Defaults to True.
            output_model_interval (int): The number of iterations between writing the
                model (and its Gibbs sampling state) to a binary file. You must also
                set the `output_model` parameter to use this option, whose argument
                will be the prefix of the filenames. Defaults to 0.
            output_state (bool): Write a compressed text file containing the words
                in the corpus with their topic assignments. The file format can easily
                be parsed and used by non-Java-based software. Defaults to True.
            output_state_interval (int): The number of iterations between writing the
                sampling state to a text file. You must also set the `output_state`
                to use this option, whose argument will be the prefix of the filenames.
                Defaults to 0.
            inference_file (bool): A topic inferencer applies a previously trained
                topic model to new documents. Defaults to False.
            evaluator_file (bool): A held-out likelihood evaluator for new documents.
                Defaults to False.
            output_topic_keys (bool): Write the top words for each topic and any
                Dirichlet parameters. Defaults to True.
            topic_word_weights_file (bool): Write unnormalized weights for every
                topic and word type. Defaults to True.
            word_topic_counts_file (bool): Write a sparse representation of topic-word
                assignments. By default this is null, indicating that no file will
                be written. Defaults to True.
            diagnostics_file (bool): Write measures of topic quality, in XML format.
                Defaults to True.
            xml_topic_report (bool): Write the top words for each topic and any
                Dirichlet parameters in XML format. Defaults to True.
            xml_topic_phrase_report (bool): Write the top words and phrases for each
                topic and any Dirichlet parameters in XML format. Defaults to True.
            output_topic_docs (bool): Currently not available. Write the most prominent
                documents for each topic, at the end of the iterations. Defaults to False.
            num_top_docs (int): Currently not available. Number of top documents for
                `output_topic_docs`. Defaults to False.
            output_doc_topics (bool): Write the topic proportions per document, at
                the end of the iterations. Defaults to True.
            doc_topics_threshold (float): Do not print topics with proportions less
                than this threshold value within `output_doc_topics`. Defaults to 0.0.
            num_topics (int): Number of topics. Defaults to 10.
            num_top_words (int): Number of keywords for each topic. Defaults to 10.
            num_interations (int): Number of iterations. Defaults to 1000.
            num_threads (int): Number of threads for parallel training.  Defaults to 1.
            num_icm_iterations (int): Number of iterations of iterated conditional
                modes (topic maximization).  Defaults to 0.
            no_inference (bool): Load a saved model and create a report. Equivalent
                to `num_iterations = 0`. Defaults to False.
            random_seed (int): Random seed for the Gibbs sampler. Defaults to 0.
            optimize_interval (int): Number of iterations between reestimating
                dirichlet hyperparameters. Defaults to 0.
            optimize_burn_in (int): Number of iterations to run before first
                estimating dirichlet hyperparameters. Defaults to 200.
            use_symmetric_alpha (bool): Only optimize the concentration parameter of
                the prior over document-topic distributions. This may reduce the
                number of very small, poorly estimated topics, but may disperse common
                words over several topics. Defaults to False.
            alpha (float): Sum over topics of smoothing over doc-topic distributions.
                alpha_k = [this value] / [num topics]. Defaults to 5.0.
            beta (float): Smoothing parameter for each topic-word. Defaults to 0.01.
            """
        self.call_mallet('train-topics', input=mallet_binary, **kwargs)
        if cleanup:
            shutil.rmtree(self.temp_output)

