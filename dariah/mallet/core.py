import logging

from .. import utils


logger = logging.getLogger(__name__)


def call(command, executable="mallet", **parameters):
    """Call MALLET.

    Parameter:
        command (str): Command for MALLET.
        executable (str): Path to MALLET executable.
        **parameter: Additional parameters for MALLET.

    Returns:
        True, if call was successful.
    """
    # Basic subprocess command:
    args = [executable, command]

    # Append additional parameters:
    for parameter, value in parameters.items():
        args.append("--{}".format(parameter.replace("_", "-")))
        if value and value != True:
            args.append(value)
    return utils.call(args)
