import logging
from pathlib import Path

from .. import utils


logger = logging.getLogger(__name__)


def call(xms="4g", jar="ddw-0.4.6.jar", **parameters):
    """Call DARIAH DKPro-Wrapper.

    Parameter:
        xms (str): Initial memory allocation pool for Java Virtual Machine.
        jar (str): Path to jarfile.
        **parameter: Additional parameters for DARIAH DKPro-Wrapper.

    Returns:
        True, if call was successful.
    """
    # Basic subprocess command:
    args = ["java", "-Xms{}".format(xms), "-jar", jar]

    # Specify path to config file:
    root = Path(jar).parent
    config = Path(root, "configs", "default.properties")
    args.extend(["-config", config])

    # Append additional parameters:
    for parameter, value in parameters.items():
        args.append("-{}".format(parameter))
        if value:
            args.append(value)
    return utils.call(args)