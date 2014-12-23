#!/usr/bin/env python

"""
Suppress output and prompt numbers in git version control.

This script will tell git to ignore prompt numbers and cell output
when looking at ipynb files if their metadata contains:

    "vc" : { "suppress_output" : true }

The notebooks themselves are not changed.

Usage instructions
==================

1. Put this script in a directory that is on the system's path.
2. Make sure it is executable by typing 
   `cd /location/of/script && chmod +x ipynb_output_filter.py`
   in a terminal.
3. Register a filter for ipython notebooks by
   creating a `.gitattributes` file in your home directory
   containing the line:
   `*.ipynb  filter=dropoutput_ipynb`
4. Connect this script to the filter by running the following 
   git commands:

   git config --global core.attributesfile ~/.gitattributes
   git config --global filter.dropoutput_ipynb.clean /path/to/script/ipynb_output_filter.py
   git config --global filter.dropoutput_ipynb.smudge cat

To enable this, open the notebook's metadata (Edit > Edit Notebook Metadata). A
panel should open containing the lines:

    {
        "name" : "",
        "signature" : "some very long hash"
    }

Add an extra line so that the metadata now looks like:

    {
        "name" : "",
        "signature" : "don't change the hash, but add a comma at the end of the line",
        "vc" : { "suppress_outputs" : true }
    }

You may need to "touch" the notebooks for git to actually register a change.

"""

import sys
import json

nb = sys.stdin.read()

json_in = json.loads(nb)
nb_metadata = json_in["metadata"]
suppress_output = False
if "vc" in nb_metadata:
    if "suppress_outputs" in nb_metadata["vc"] and nb_metadata["vc"]["suppress_outputs"]:
        suppress_output = True
if not suppress_output:
    sys.stdout.write(nb)
    exit() 


ipy_version = int(json_in["nbformat"])-1 # nbformat is 1 more than actual version.

def strip_output_from_cell(cell):
    if "outputs" in cell:
        cell["outputs"] = []
    if "prompt_number" in cell:
        del cell["prompt_number"]


if ipy_version == 2:
    for sheet in json_in["worksheets"]:
        for cell in sheet["cells"]:
            strip_output_from_cell(cell)
else:
    for cell in json_in["cells"]:
        strip_output_from_cell(cell)

json.dump(json_in, sys.stdout, sort_keys=True, indent=1, separators=(",",": "))
