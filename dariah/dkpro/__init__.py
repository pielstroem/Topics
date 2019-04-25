"""
This module allows you to communicate with the Java CLI tool
DARIAH DKPro-Wrapper in Python:

.. code-block:: python
   dkpro = dariah.dkpro.api.DKPro(jar="ddw-0.4.6.jar")
   dkpro.process(directory="corpus",
                 output="annotated")
"""

from .processing import DKPro
