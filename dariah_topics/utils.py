"""
Utilizing the Command-Line from Within the Python Environment
*************************************************************

Functions of this module are for **utilizing the command-line**. You can call \
the command-line from within Python using :func:`call_commandline()`, e.g. the \
`DARIAH-DKPro-Wrapper <https://github.com/DARIAH-DE/DARIAH-DKPro-Wrapper>`_ for \
processing and annotating text corpora. Using :class:`Mallet`, you can call the \
NLP-tool MALLET.

Contents
********
    * :func:`call_commandline()` calls based on the elements of a list the command-\
        line.
    * :class:`Mallet` is a class containing methods to call the NLP-tool MALLET.
    * :meth:`call_mallet()` calls MALLET with a specific executable and additional \
        parameteres.
    * :meth:`import_corpus()` imports a text corpus to the specific MALLET corpus \
        format. Uses the executable ``import-dir``.
    * :meth:`train_topics()` creates a topic model with the imported text corpus. \
        Uses the executable ``train-topics``.

"""

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

log = logging.getLogger('dariah_topics')


def _decode(std):
    """Decodes the bytes-like output of a subprocess in UTF-8.
    
    This private function is wrapped in :func:`call_commandline()`.
    
    Args:
        std (bytes-like): The ``stdout``  or ``stderr`` (or whatever) of a
            subprocess.
        
    Returns:
        A list of decoded strings.
        
    Example:
        >>> _decode([bytes('This is a test.', encoding='utf-8')])
        ['This is a test.']
    """
    return [line.decode('utf-8').replace('\n', '') for line in std]


def call_commandline(cmd, stdin=None, stdout='pipe', stderr='pipe', communicate=False, logfile=False):
    """Calls the command-line from within Python.
    
    With this function you can call the command-line with a specific command. Each \
    argument has to be an element in a list (``cmd``).
    
    Args:
        cmd (list): A list of command-line arguments.
        stdin (str), optional: Value for stdin. Defaults to None.
        stdout (str), optional: Value for stdout. Defaults to ``pipe``.
        stderr (str), optional: Value for stderr. Defaults to ``pipe``.
        communicate (bool), optioanl: If True, ``stdout`` and ``stderr`` will be
            processed. Defaults to False.
        logfile (bool), optional: If True, a logfile (``commandline.log``) will
            be created. Otherwise ``stdout`` (and ``stderr``, respectively) will
            be printed as logging to the console (level: INFO).
        
    Returns:
        :class:`Popen` object of the subprocess.
        
    Example:
        >>> call_commandline(['python', '-h']) # doctest: +ELLIPSIS
        <subprocess.Popen object at ...>
    """
    if stdin == 'pipe':
        stdin = PIPE
    if stdout == 'pipe':
        stdout = PIPE
    if stderr == 'pipe':
        stderr = PIPE
    
    if not all(isinstance(arg, str) for arg in cmd):
        cmd = [str(arg) for arg in cmd]
    
    log.info("Calling the command-line: {0} ...".format(' '.join(cmd)))

    process = Popen(cmd, stdin=stdin, stdout=stdout, stderr=stderr)
    decoded_stderr = _decode(process.stderr)

    if communicate:
        decoded_stderr = _decode(process.stderr)
        decoded_stdout = _decode(process.stdout)
        if logfile:
            log.info("Check commandline.log in '{0}' for logging.".format(os.getcwd()))
            with open('commandline.log', 'w', encoding='utf-8') as file:
                file.write('\n'.join(decoded_stdout))
                file.write('\n'.join(decoded_stderr))
        else:
            for line_stdout in decoded_stdout:
                log.info(line_stdout)
            for line_stdout in decoded_stderr:
                log.info(line_stderr)
    return process


def _check_whitespace(string):
    """Checks if whitespaces are in a string.
    
    This private function is wrapped in :func:`call_mallet()`.
    
    Args:
        string (str): Obviously, a string.
        
    Returns:
        True if whitespaces are **not** in the string, otherwise False.
        
    Example:
        >>> _check_whitespace('nowhitespace')
        True
        >>> _check_whitespace('white space')
        False
    """
    if not re.search(r'\s', str(string)):
        return True
    else:
        return False


def _check_mallet_output(keyword, kwargs=None):
    """Checks if MALLET created output.
    
    This private function is wrapped in :func:`import_tokenized_corpus()` and \
    :func:`train_topics()`.
    
    Args:
        keyword (str): A token, which has to be in ``kwargs.values()``.
        kwargs (dict), optional: Args for the MALLET functions.
    
    Raises:
        OSError, if MALLET did not produce any output files.
    """
    if not 'corpus.mallet' in keyword:
        output_files = [value for arg, value in kwargs.items() if keyword in arg or 'txt' in str(value) or 'xml' in str(value)]
    else:
        output_files = [keyword]

    if not all(os.path.exists(file) for file in output_files):
        raise OSError("MALLET did not produce any output files. Maybe check your args?")


class Mallet:
    """Python wrapper for MALLET.
    
    With this class you can call the command-line tool `MALLET <http://mallet.cs.umass.edu/topics.php>`_ \
    from within Python.
    """
    def __init__(self, executable='mallet', corpus_output=None, logfile=False):
        self.executable = shutil.which(executable)
        if self.executable is None:
            raise FileNotFoundError(("The executable '{0}' could not be found.\n"
                                     "Either place the executable into the $PATH or call "
                                     "{1}(executable='/path/to/mallet')").format(executable, self.__class__.__name__))
        if corpus_output is None:
            self.corpus_output = tempfile.mkdtemp()
        else:
            self.corpus_output = corpus_output
        self.logfile = logfile

    def call_mallet(self, command, **kwargs):
        """Calls the command-line tool MALLET.
        
        With this function you can call `MALLET <http://mallet.cs.umass.edu/topics.php>`_ \
        using a specific ``command`` (e.g. ``train-topics``) and its parameters.
        **Whitespaces (especially for Windows users) are not allowed in paths.**
        
        Args:
            command (str): A MALLET command, this could be ``import-dir`` (load
                the contents of a directory into MALLET instances), ``import-file``
                (load a single file into MALLET instances), ``import-svmlight``
                (load SVMLight format data files into MALLET instances), ``info``
                (get information about MALLET instances), ``train-classifier``
                (train a classifier from MALLET data files), ``classify-dir``
                (classify data from a single file with a saved classifier), ``classify-file``
                (classify the contents of a directory with a saved classifier),
                ``classify-svmlight`` (classify data from a single file in SVMLight
                format), ``train-topics`` (train a topic model from MALLET data
                files), ``infer-topics`` (use a trained topic model to infer topics
                for new documents), ``evaluate-topics`` (estimate the probability
                of new documents under a trained model), ``prune`` (remove features
                based on frequency or information gain), ``split`` (divide data
                into testing, training, and validation portions), ``bulk-load``
                (for big input files, efficiently prune vocabulary and import docs).

        Returns:
            :class:`Popen` object of the MALLET subprocess.
            
        Example:
            >>> import tempfile
            >>> with tempfile.NamedTemporaryFile(suffix='.txt') as tmpfile:
            ...     tmpfile.write(b"This is a plain text example.") and True
            ...     tmpfile.flush()
            ...     Mallet = Mallet(corpus_output='.')
            ...     process = Mallet.call_mallet('import-file', input=tmpfile.name)
            ...     os.path.exists('text.vectors')
            True
            True
            """
        args = [self.executable, command]
        for option, value in kwargs.items():
            args.append('--' + option.replace('_', '-'))
            if value is not None:
                 args.append(value)
                 
        if not all(_check_whitespace(arg) for arg in args):
            raise ValueError("Whitespaces are not allowed in '{0}'".format(args))
            
        if self.logfile:
            communicate = True
        else:
            communicate = False
        
        return call_commandline(args, communicate=communicate, logfile=self.logfile)

    def import_tokenized_corpus(self, tokenized_corpus, document_labels, **kwargs):
        """Creates MALLET corpus model.
        
        With this function you can import a ``tokenized_corpus`` to create the \
        MALLET corpus model. The MALLET command for this step is ``import-dir`` \
        with ``--keep-sequence`` (which is already defined in the function, so \
        you don't have to), but you have the ability to specify all available \
        parameters. The output will be saved in ``output_corpus``.
        
        Args:
            tokenized_corpus (list): Tokenized corpus containing one or more
                iterables containing tokens.
            document_labels (list): Name of each `tokenized_document` in `tokenized_corpus`.
            encoding (str): Character encoding for input file. Defaults to UTF-8.
            token_regex (str): Divides documents into tokens using a regular
                expression (supports Unicode regex). Defaults to \p{L}[\p{L}\p{P}]+\p{L}.
            preserve_case (bool): If False, converts all word features to lowercase.
                Defaults to False.
            remove_stopwords (bool): Ignores a standard list of very common English
                tokens. Defaults to True.
            stoplist (str): Absolute path to plain text stopword list. Defaults to None.
            extra_stopwords (str): Read whitespace-separated words from this file,
                and add them to either the default English stoplist or the list
                specified by ``stoplist``. Defaults to None.
            stop_pattern_file (str): Read regular expressions from a file, one per
                line. Tokens matching these regexps will be removed. Defaults to None.
            skip_header (bool): If True, in each document, remove text occurring
                before a blank line. This is useful for removing email or UseNet
                headers. Defaults to False.
            skip_html (bool): If True, remove text occurring inside <...>, as in
                HTML or SGML. Defaults to False.
            replacement_files (str): Files containing string replacements, one per
                line: 'A B [tab] C' replaces A B with C, 'A B' replaces A B with A_B.
                Defaults to None.
            deletion_files (str): Files containing strings to delete after
                `replacements_files` but before tokenization (i.e. multiword stop
                terms). Defaults to False.
            keep_sequence_bigrams (bool): If True, final data will be a
                FeatureSequenceWithBigrams rather than a FeatureVector. Defaults to False.
            binary_features (bool): If True, features will be binary. Defaults to False.
            save_text_in_source (bool): If True, save original text of document in source.
                Defaults to False.
            print_output (bool): If True, print a representation of the processed data
                to standard output. This option is intended for debugging. Defaults to
                False.

        Returns:
            The absolute path to the created MALLET corpus file.
            
        Example:
            >>> tokenized_corpus = [['this', 'is', 'a', 'tokenized', 'document']]
            >>> document_labels = ['document_label']
            >>> Mallet = Mallet(corpus_output='.')
            >>> mallet_corpus = Mallet.import_tokenized_corpus(tokenized_corpus, document_labels)
            >>> os.path.exists('corpus.mallet')
            True
        """
        corpus_file = os.path.join(self.corpus_output, 'corpus.mallet')
        postprocessing.save_tokenized_corpus(tokenized_corpus, document_labels, self.corpus_output)
        self.call_mallet('import-dir', keep_sequence=None, input=self.corpus_output, output=corpus_file, **kwargs)
        
        _check_mallet_output(os.path.join(self.corpus_output, 'corpus.mallet'))  
        
        return corpus_file

    def train_topics(self, mallet_binary, cleanup=False, **kwargs):
        """Trains LDA model.
        
        With this function you can train a topic model. The MALLET command for \
        this step is ``train-topics`` (which is already defined in the function, \
        so you don't have to), but you have the ability to specify all available \
        parameters.
        
        Args:
            mallet_binary (str): Path to MALLET corpus model.
            cleanup (bool): If True, the directory ``corpus_output`` will be removed
                after modeling.
            input_model (str): The filename from which to read the binary topic
                model.
            input_state (str): The filename from which to read the gzipped Gibbs
                sampling state created by ``output_state``. The original input
                file must be included, using ``input``.
            output_model (str): The filename in which to write the binary topic
                model at the end of the iterations.
            output_state (str): The filename in which to write the Gibbs sampling
                state after at the end of the iterations.
            output_model_interval (int): The number of iterations between writing
                the model (and its Gibbs sampling state) to a binary file. You must
                also set the ``output_model`` to use this option, whose argument
                will be the prefix of the filenames. Default is 0.
            output_state_interval (int): The number of iterations between
                writing the sampling state to a text file. You must also set
                the ``output_state`` to use this option, whose argument will be
                the prefix of the filenames. Default is 0.
            inferencer_filename (str): A topic inferencer applies a previously
                trained topic model to new documents.
            evaluator_filename (str): A held-out likelihood evaluator for new documents.
            output_topic_keys (str): The filename in which to write the top
                words for each topic and any Dirichlet parameters.
            num_top_words (int): The number of most probable words to print for
                each topic after model estimation. Default is 20.
            show_topics_interval (int): The number of iterations between printing
                a brief summary of the topics so far. Default is 50.
            topic_word_weights_file (str): The filename in which to write
                unnormalized weights for every topic and word type.
            word_topic_counts_file (str): The filename in which to write a sparse
                representation of topic-word assignments.
            diagnostics_file (str): The filename in which to write measures of
                topic quality, in XML format.
              Default is null
            xml_topic_report (str): The filename in which to write the top words
                for each topic and any Dirichlet parameters in XML format.
            xml_topic_phrase_report (str): The filename in which to write the top
                words and phrases for each topic and any Dirichlet parameters in
                XML format.
            num_top_docs (int): When writing topic documents with ``output_topic_docs``,
                report this number of top documents. Default is 100
            output_doc_topics (str): The filename in which to write the topic
                proportions per document, at the end of the iterations.
            doc_topics_threshold (float): Do not print topics with proportions less
                than this threshold value within ``output_doc_topics``. Defaults to 0.0.
            num_topics (int): Number of topics. Defaults to 10.
            num_top_words (int): Number of keywords for each topic. Defaults to 10.
            num_interations (int): Number of iterations. Defaults to 1000.
            num_threads (int): Number of threads for parallel training.  Defaults to 1.
            num_icm_iterations (int): Number of iterations of iterated conditional
                modes (topic maximization).  Defaults to 0.
            no_inference (bool): Load a saved model and create a report. Equivalent
                to ``num_iterations = 0``. Defaults to False.
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
                ``alpha_k = [this value] / [num topics]``. Defaults to 5.0.
            beta (float): Smoothing parameter for each topic-word. Defaults to 0.01.
            
        Returns:
            None.
            
        Example:
            >>> tokenized_corpus = [['this', 'is', 'a', 'tokenized', 'document']]
            >>> document_labels = ['document_label']
            >>> Mallet = Mallet(corpus_output='.')
            >>> mallet_corpus = Mallet.import_tokenized_corpus(tokenized_corpus, document_labels)
            >>> mallet_topics = Mallet.train_topics(mallet_corpus,
            ...                                     output_model='model.mallet',
            ...                                     num_iterations=10)
            >>> os.path.exists('model.mallet')
            True
        """
        self.call_mallet('train-topics', input=mallet_binary, **kwargs)
        
        _check_mallet_output('output', kwargs)

        if cleanup:
            shutil.rmtree(self.corpus_output)

