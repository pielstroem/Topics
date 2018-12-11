r"""
dariah.dkpro.core
~~~~~~~~~

This module implements the core functions of the DKPro module.
"""

from pathlib import Path

from .. import utils


def call(jar, xms="4g", **parameters):
    r"""Call DARIAH DKPro-Wrapper.

    Parameter:
        xms (str): Initial memory allocation pool for Java Virtual Machine.
        jar (str): Path to jarfile.
        **parameter: Additional parameters for DARIAH DKPro-Wrapper.

    Returns:
        True, if call was successful.
    """
    # Basic subprocess command:
    args = ["java", "-Xms{}".format(xms), "-jar", jar]

    # Append additional parameters:
    for parameter, value in parameters.items():
        # Support synonyms for `-input` parameter:
        if parameter in {"filepath", "directory", "path"}:
            args.append("-input")
        else:
            args.append("-{}".format(parameter))
        if value:
            args.append(str(value))
    return utils.call(args)
