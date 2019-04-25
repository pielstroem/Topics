"""
This module allows you to communicate with the Java CLI tool
MALLET:

.. code-block:: python
   mallet = dariah.mallet.api.MALLET(executable="mallet")
   mallet.import_dir(directory="corpus",
                     output="corpus.mallet")
"""

from .api import MALLET


# TODO ABSOLUTE PFADE FÜR DATEIEN
