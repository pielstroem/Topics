# Topics - Easy Topic Modeling in Python #

Topics is a gently introduction to Topic Modeling. It provides a convientient, modular workflow that can be entirely controlled from within and which comes with a well documented [Jupyter notebook](http://jupyter.org/), integreating the two of the most popular LDA implementations: [Gensim](https://radimrehurek.com/gensim/) and [Mallet](http://mallet.cs.umass.edu/). Users not yet familiar with working with Python scripts can test basic topic modeling in a [Flask](http://flask.pocoo.org/)-based [GUI demonstrator](/demonstrator/README.md).

### Getting Started

#### Windows

1. Download and install the latest version of [WinPython](https://winpython.github.io/), Note: For older WinPython versions, Windows 7/8 users may have to install [Microsoft Visual C++ Redistributable for Visual Studio 2015](https://www.microsoft.com/de-de/download/details.aspx?id=48145)
2. Download and install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
3. Open the [command-line](https://en.wikipedia.org/wiki/Command-line_interface), navigate to your **WinPython** folder via `cd C:\\path-to-your-folder` and type `git clone https://github.com/DARIAH-DE/Topics.git` to clone **Topics** into your WinPython folder
4. Open the [Winpython Command Prompt](https://github.com/winpython/winpython/wiki/Installing-Additional-Packages) and type `easy_install -U gensim` to install [Gensim](https://radimrehurek.com/gensim/)
5. Open the [Winpython Command Prompt](https://github.com/winpython/winpython/wiki/Installing-Additional-Packages), navigate to the **Topics** folder and type `pip install .` to install packages required by 'Topics'
6. Access the folder **Topics** through [Jupyter](in your WinPython folder) in your browser, open the [Introduction.ipynb](Introduction.ipynb) and follow the instructions


#### Unix/Linux

1. Download and install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git), 
2. Open the [command-line interface](https://en.wikipedia.org/wiki/Command-line_interface), type `git clone https://github.com/DARIAH-DE/Topics.git` and press Enter
3. Note: The distribution packages 'libfreetype6-dev' and 'libpng-dev' and a compiler for c++, e.g [gcc](https://gcc.gnu.org/) have to be installed 
4. Open the [command-line interface](https://en.wikipedia.org/wiki/Command-line_interface), navigate to the folder **Topics**  and type `pip install . --user` to install the required packages 
5. Install [Jupyter](http://jupyter.readthedocs.io/en/latest/install.html) and run it by typing `jupyter notebook` in the command-line
5. Access the folder **Topics** through Jupyter in your browser, open the [Introduction.ipynb](Introduction.ipynb) and follow the instructions

