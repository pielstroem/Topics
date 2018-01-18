# Topics – Easy Topic Modeling in Python
[Topics](http://dev.digital-humanities.de/ci/job/DARIAH-Topics/doclinks/1/) is a Python library for Topic Modeling. Furthermore, this repository provides a convenient, modular workflow that can be entirely controlled from within a well documented [Jupyter](http://jupyter.org/) notebook. Users not yet familiar with programming in Python can test basic Topic Modeling in a [Flask](http://flask.pocoo.org/)-based [GUI demonstrator](/demonstrator/README.md). **For a standalone application**, which does not require a Python interpreter or any extra installations, **have a look at the [release-section](https://github.com/DARIAH-DE/Topics/releases)**.

At the moment, this library supports three LDA implementations:
* [lda](http://pythonhosted.org/lda/index.html), which is lightweight and provides basic LDA.
* [MALLET](http://mallet.cs.umass.edu/), which is known to be very robust.
* [Gensim](https://radimrehurek.com/gensim/), which is attractive because of its multi-core support.

## Resources
* [Topics website](http://dev.digital-humanities.de/ci/job/DARIAH-Topics/doclinks/1/)
* [Topics API documentation](http://dev.digital-humanities.de/ci/job/DARIAH-Topics/doclinks/1/docs/gen/modules.html)
* [Topics paper](https://dh2017.adho.org/abstracts/411/411.pdf)
* **[Standalone Demonstrator releases](https://github.com/DARIAH-DE/Topics/releases)**
* [An introduction to Topic Modeling using lda](IntroducingLda.ipynb)
* [An introduction to Topic Modeling using MALLET](IntroducingMallet.ipynb)
* [An introduction to Topic Modeling using Gensim](IntroducingGensim.ipynb)

## Installation
To install the latest stable version of the library `dariah_topics`:

```
$ pip install git+https://github.com/DARIAH-DE/Topics.git
```

To install the latest development version:

```
$ pip install --upgrade git+https://github.com/DARIAH-DE/Topics.git@testing
```

## Working with Jupyter Notebooks
If you wish to work through the tutorials, you can clone the repository using [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git):

```
$ git clone https://github.com/DARIAH-DE/Topics.git
```

or download the [ZIP-archive](https://github.com/DARIAH-DE/Topics/archive/master.zip) (don't forget to unzip it) and install `dariah_topics` from its source code:

```
$ python setup.py install
```

As a server-client application, Jupyter allows you to edit and run Python code interactively from within so-called notebooks via a web browser.

To install Jupyter:

```
$ pip install jupyter
```

> Python distributions like [Anaconda](https://anaconda.org/anaconda/python) come with Jupyter by default.

You can run Jupyter via:

```
$ jupyter notebook
```

## Working with MALLET
[MALLET](http://mallet.cs.umass.edu) is a Java-based package for statistical *natural language processing*. The MALLET Topic Model package includes an extremely fast and highly scalable implementation of Gibbs sampling and tools for inferring topics for new documents given trained models.

To call MALLET from within the Python environment, `dariah_topics` provides a convenient wrapper.

You can download MALLET [here](http://mallet.cs.umass.edu/download.php). For more detailed instructions, have a look at [this](http://programminghistorian.org/lessons/topic-modeling-and-mallet).

## Troubleshooting
If you are confronted with any issues regarding installation or usability, please use [GitHub issues](https://github.com/DARIAH-DE/Topics/issues).

**This library requires Python 3.6 or higher.**

### Windows-specific Issues
* You will have to install `future‑0.16.0‑py3‑none‑any.whl` from [this resource](http://www.lfd.uci.edu/~gohlke/pythonlibs/). Download the appropriate file and run `pip install future‑0.16.0‑py3‑none‑any.whl`.
* In case of the error `Microsoft Visual C++ 10.0 is required`, check if you are using Python 3.6 or higher with `python -V`. If you do, you have to install Microsoft Windows SDK from [this resource](www.microsoft.com/download/details.aspx?id=8279). If you do not, upgrade to Python 3.6 or higher and try installing the library again.

### UNIX-specific Issues
* In case of `PermissionError: [Errno 13] Permission denied`, try `pip install --user` or `python setup.py install --user`, respectively.
* Due to several visualization dependencies, you might have to install the distribution packages `libfreetype6-dev` and `libpng-dev` (e.g. using `sudo apt-get install`).

### MacOS-specific Issues
* Make sure to install Python 3.6 correctly and adjust the selection of the Python interpreter in your editor accordingly. See also: https://docs.python.org/3/using/mac.html