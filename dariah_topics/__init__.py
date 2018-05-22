"""
The :mod:`dariah_topics` package currently offers seven modules:

* :mod:`dariah_topics.evaluation` for evaluating semantic coherence of topics.
* :mod:`dariah_topics.postprocessing` for postprocessing text data.
* :mod:`dariah_topics.preprocessing` for preprocessing text data.
* :mod:`dariah_topics.utils` for some useful command-line utils.
* :mod:`dariah_topics.visualization` for visualizing the output of LDA models.
* :mod:`dariah_topics.modeling` is a wrapper for the LDA implementations.
"""

from dariah_topics import evaluation
from dariah_topics import postprocessing
from dariah_topics import preprocessing
from dariah_topics import utils
from dariah_topics import visualization
from dariah_topics import modeling

__author__ = "Sina Bock, Philip Duerholt, Michael Huber, Thora Hagen, Severin Simmler, Thorsten Vitt"
__version__ = "1.0"
__copyright__ = "Copyright 2017-2018, DARIAH-DE"
__credits__ = ["Sina Bock", "Philip Duerholt", "Michael Huber", "Thora Hagen", "Steffen Pielstroem", "Severin Simmler", "Thorsten Vitt"]
__maintainer__ = "Steffen Pielstroem"
__email__ = "pielstroem@biozentrum.uni-wuerzburg.de"
__status__ = "Development"
