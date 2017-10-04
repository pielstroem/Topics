#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='dariah_topics',
    version='0.3.0dev1',
    description='DARIAH Topic Modelling',
    # url
    author="DARIAH-DE Wuerzburg Group",
    author_email="pielstroem@biozentrum.uni-wuerzburg.de",
    # license
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    # keywords
    packages=find_packages(exclude=['corpora', 'demonstrator', 'test', 'tutorial_supplementals']),
    install_requires=[
        'pandas>=0.19.2',
        'regex>=2017.01.14',
        'gensim>=0.13.2',
        'lda>=1.0.5',
        'numpy>=1.3',
        'lxml>=3.6.4'
    ],
    # pip install -e .[demonstrator,vis,evaluation]
    extras_require={
        'demonstrator': [
            'flask>=0.11.1'
        ],
        'vis': [
            'matplotlib>=1.5.3',
            'bokeh>=0.12.6'
            #'wordcloud>=1.3.1'
        ]
    }
)
