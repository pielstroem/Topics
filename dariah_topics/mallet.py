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


def call_commandline(cmd, logfile=False, stdin=None, stdout='pipe', stderr='pipe', communicate=True):
    if stdin == 'pipe':
        stdin = PIPE
    if stdout == 'pipe':
        stdout = PIPE
    if stderr == 'pipe':
        stderr = PIPE
    
    cmd = [str(arg) for arg in cmd]
    log.info("Calling the command-line with {} ...".format(' '.join(cmd)))
    log.debug("stdin = {}".format(stdin))
    log.debug("stdout = {}".format(stdout))
    log.debug("stderr = {}".format(stderr))

    p = Popen(cmd, stdin=stdin, stdout=stdout, stderr=stderr)
    decoded_stderr = _decode_stdout(p.stderr)

    if communicate:
        if logfile:
            log.info("Check mallet.log in {} for logging.".format(os.getcwd()))
            with open('mallet.log', 'w', encoding='utf-8') as file:
                file.write('\n'.join(decoded_stderr))
        else:
            [log.debug(line) for line in decoded_stderr]
    elif p.returncode != 0:
        raise OSError(decoded_stderr)
    else:
        decoded_stdout = _decode_stdout(p.stdout)
        log.debug(decoded_stdout)
    return None


def _check_whitespace(string):
    if not re.search(r'\s', str(string)):
        return True
    else:
        return False


class Mallet:
    def __init__(self, executable='mallet', temp_output=None, logfile=True):
        self.executable = shutil.which(executable)
        if self.executable is None:
            raise FileNotFoundError(("The executable '{0}' could not be found.\n"
                                     "Either place the executable into the $PATH or call "
                                     "{1}(executable='/path/to/mallet')").format(executable, self.__class__.__name__))
        if temp_output is None:
            prefix = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(5)])
            temp_output = os.path.join(tempfile.gettempdir(), prefix)
        self.temp_output = temp_output
        self.logfile = logfile

    def call_mallet(self, command, **kwargs):
        args = [self.executable, command]
        for option, value in kwargs.items():
            args.append('--' + option.replace('_', '-'))
            if value is not None:
                 args.append(value)
        if not all(_check_whitespace(arg) for arg in args):
            raise ValueError("Whitespaces are not allowed in {}".format(args))
        return call_commandline(args, self.logfile)

    def import_tokenized_corpus(self, tokenized_corpus, document_labels, **kwargs):
        mallet_binary = os.path.join(self.temp_output, 'corpus.mallet')
        postprocessing.save_tokenized_corpus(tokenized_corpus, document_labels, self.temp_output)
        self.call_mallet('import-dir', keep_sequence=None, input=self.temp_output, output=mallet_binary, **kwargs)
        return mallet_binary

    def train_topics(self, mallet_binary, cleanup=True, **kwargs):
        self.call_mallet('train-topics', input=mallet_binary, **kwargs)
        if cleanup:
            shutil.rmtree(self.temp_output)

