import logging
import subprocess


logger = logging.getLogger(__name__)


def call(args):
    """Call a subprocess.

    Parameter:
        args (list): The subprocess’ arguments.

    Returns:
        True, if call was successful.
    """
    for message in _process(args):
        # No logging, just printing, because this is
        # not related to Python.
        print(message)
    return True


def _process(args):
    """Construct a process.
    """
    popen = subprocess.Popen(args,
                             stdout=subprocess.PIPE,
                             universal_newlines=True)
    # Yield every line of stdout:
    for line in iter(popen.stdout.readline, ""):
        yield line
    popen.stdout.close()
    code = popen.wait()
    if code:
        raise subprocess.CalledProcessError(code, args)
