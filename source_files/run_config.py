# -*- coding:Utf-8 -*-
"""Input-file overrides for the batch runner run.py (port only).

Actions 6, 7 and 9 copy a fixed top-level input file into their output folder
(e.g. '6_simulations_MH_to_compare.py').  run.py can point them at another
file by filling `overrides`; with no override (main_script.py) the original
file name is used, so behaviour is unchanged.
"""

overrides = {}


def source(default_name):
    """Path of the input file to use in place of `default_name`."""
    return overrides.get(default_name, default_name)
