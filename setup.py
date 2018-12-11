#!/usr/bin/env python3

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

NAME = "dariah"
DESCRIPTION = "A library for natural language processing, topic modeling, and visualization."
AUTHOR = "DARIAH-DE"
EMAIL = "pielstroem@biozentrum.uni-wuerzburg.de"
URL = "https://dariah-de.github.io/Topics"
REQUIRES_PYTHON = ">=3.4.0"
VERSION = None
REQUIRED = [
    ""
]

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

about = {}
if not VERSION:
    with open(os.path.join(here, NAME, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


class UploadCommand(Command):
    """Support setup.py upload.
    """

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds ...")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building source and wheel distribution ...")
        os.system("{0} setup.py sdist bdist_wheel".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine ...")
        os.system("twine upload --repository-url https://upload.pypi.org/legacy/ dist/*")

        self.status("Pushing git tags ...")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")
        
        sys.exit()

setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["docs", "test", "notebooks"]),
    install_requires=REQUIRED,
    include_package_data=True,
    license="Apache 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    cmdclass={
        "upload": UploadCommand,
    }
)
