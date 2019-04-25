import os
import shutil
import sys
import setuptools


NAME = "dariah"
DESCRIPTION = "A library for topic modeling and visualization."
URL = "https://dariah-de.github.io/Topics"
AUTHOR = "DARIAH-DE"
REQUIRES_PYTHON = ">=3.6.0"
VERSION = "2.0.0"
REQUIRED = ["regex>=2019.4.12",
            "pandas>=0.24.2",
            "numpy>=1.16.2",
            "lda>=1.1.0",
            "matplotlib>=3.0.3",
            "cophi>=1.3.2",
            "seaborn>=0.9.0"]


with open("README.md", "r", encoding="utf-8") as readme:
    long_description = f"\n{readme.read()}"


class UploadCommand(setuptools.Command):
    description = "Build and publish the package."
    user_options = list()

    @staticmethod
    def status(s):
        print(f"\033[1m{s}\033[0m")

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds...")
            shutil.rmtree("dist")
        except OSError:
            pass

        self.status("Building source and wheel distribution...")
        os.system(f"{sys.executable} setup.py sdist bdist_wheel")

        self.status("Uploading the package to PyPI via Twine...")
        os.system("twine upload dist/*")

        self.status("Pushing git tags...")
        os.system(f"git tag v{VERSION}")
        os.system("git push --tags")

        sys.exit()


setuptools.setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=REQUIRED,
    include_package_data=True,
    license="Apache 2.0",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7"
    ],
    cmdclass={
        "upload": UploadCommand,
    },
)
