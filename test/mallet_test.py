from nose.tools import raises
from dariah_topics.mallet import Mallet

@raises(FileNotFoundError)
def command_not_found_test():
    """When the mallet executable was not found, raise an exception."""
    Mallet(executable="i_am_an_executable_that_does_not_exist")
