#!/usr/bin/env python3

from setuptools import setup, find_packages

PROJECT = 'DARIAH Topic Modeling'
VERSION = '1.0'
REVISION = '1.0.0.dev'
AUTHOR = 'DARIAH-DE Wuerzburg Group'
AUTHOR_EMAIL = 'pielstroem@biozentrum.uni-wuerzburg.de'
URL = 'https://dariah-de.github.io/Topics'

setup(
    name='dariah_topics',
    version=REVISION,
    description=PROJECT,
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    keywords=['topic modeling', 'lda', 'natural language processing', 'digital humanities'],
    packages=find_packages(exclude=['docs', 'test', 'notebooks']),
    install_requires=[
        'pandas>=0.19.2',
        'regex>=2017.01.14',
        'gensim>=0.13.2',
        'lda>=1.0.5',
        'numpy>=1.3',
        'lxml>=3.6.4',
        'matplotlib>=1.5.3',
        'bokeh>=0.12.6',
        'metadata_toolbox'
    ],
    command_options={
        'build_sphinx': {
            'project': ('setup.py', PROJECT),
            'version': ('setup.py', VERSION),
            'release': ('setup.py', REVISION),
        }
    }
)
