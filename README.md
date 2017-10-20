# Topics – Easy Topic Modeling in Python

[Topics](http://dev.digital-humanities.de/ci/job/DARIAH-Topics/doclinks/1/) is a Python library for Text Mining and Topic Modeling. Furthermore, this repository provides a convenient, modular workflow that can be entirely controlled from within and which comes with a well documented [Jupyter](http://jupyter.org/) notebook. Users not yet familiar with programming in Python can test basic Topic Modeling in a [Flask](http://flask.pocoo.org/)-based [GUI demonstrator](/demonstrator/README.md). For a standalone application, which does not require a Python interpreter or any extra installations, have a look at the [release-section](https://github.com/DARIAH-DE/Topics/releases).

At the moment, this library supports three LDA implementations:
* [lda](http://pythonhosted.org/lda/index.html), which is lightweight and provides basic LDA.
* [MALLET](http://mallet.cs.umass.edu/), which is known to be very robust.
* [Gensim](https://radimrehurek.com/gensim/), which is attractive because of its multi-core support.

## Resources
* [Topics website](http://dev.digital-humanities.de/ci/job/DARIAH-Topics/doclinks/1/)
* [Topics API documentation](http://dev.digital-humanities.de/ci/job/DARIAH-Topics/doclinks/1/docs/gen/modules.html)
* [Topics paper](https://dh2017.adho.org/abstracts/411/411.pdf)
* [Demonstrator releases](https://github.com/DARIAH-DE/Topics/releases)
* [An introduction to Topic Modeling using lda](IntroducingLda.ipynb)
* [An introduction to Topic Modeling using MALLET](IntroducingMallet.ipynb)
* [An introduction to Topic Modeling using Gensim](IntroducingGensim.ipynb)

### Getting Started
#### Windows
1.  Download and install the latest version of [WinPython](https://winpython.github.io/).
2.  Download and install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
3.  Open the **WinPython PowerShell Prompt.exe** in your **WinPython** folder and type `git clone https://github.com/DARIAH-DE/Topics.git` to clone **Topics** into your WinPython folder.
4.  Type `cd .\Topics` in **WinPython PowerShell** to navigate to the **Topics** folder. 
5. Either: Type `pip install .` in **WinPython PowerShell** to install packages required by **Topics** 
5. Or: Type `pip install -r requirements.txt` in **Winpython PowerShell** to install **Topics** with additional development packages.
6.  Type `jupyter notebook` in **WinPython PowerShell** to open Jupyter, select one of the files with suffix `.ipynb` and follow the instructions.
7.  **Note**: For the development packages the Python module **future** is needed. Depending in your WinPython and your Windows version you might have to install **future** manually.
8.  Therefore, download the latest [future-x.xx.x-py3-none-any.whl](http://www.lfd.uci.edu/~gohlke/pythonlibs/).
9.  Open the **WinPython Control Panel.exe** in your **WinPython** folder.
10. Install the **future**-wheel via the **WinPython Control Panel.exe**.

11. **Troubleshooting**: If the installing process fails and you get the error message: Microsoft Visual C++ 10.0 is required please check if you are using python 3.6. (Type 'python -V')


#### macOS and Linux
1. Download and install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
2. Open the [command-line interface](https://en.wikipedia.org/wiki/Command-line_interface), type `git clone https://github.com/DARIAH-DE/Topics.git` to clone **Topics** into your working directory.
3. **Note**: The distribution packages `libfreetype6-dev` and `libpng-dev` and a compiler for C++, e.g. [gcc](https://gcc.gnu.org/) have to be installed.
4. Open the [command-line interface](https://en.wikipedia.org/wiki/Command-line_interface), navigate to the folder **Topics**  and type `pip install . --user` to install the required packages.
5. Install [Jupyter](http://jupyter.readthedocs.io/en/latest/install.html) and run it by typing `jupyter notebook` in the command-line.
5. Access the folder **Topics** through Jupyter in your browser, select one of the files with suffix `.ipynb` and follow the instructions.


#### Working with MALLET
1. Download and unzip [MALLET](http://mallet.cs.umass.edu).
2. Set the environment variable for MALLET.


For more detailed instructions, have a look at [this](http://programminghistorian.org/lessons/topic-modeling-and-mallet).
