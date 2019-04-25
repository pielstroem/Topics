"""
dariah.topics.utils
~~~~~~~~~~~~~~~~~~~

This module implements helper functions for topic modeling.
"""

import cophi

def read_topics_file(path):
    with path.open("r", encoding="utf-8") as file:
        for row in file:
            sequence = row.split("\t")[2]
            yield list(cophi.utils.find_tokens(sequence))[:200]
