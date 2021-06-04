import os

rdir = os.path.abspath(__file__)
assert rdir.endswith("xaio_config.py")
root_dir = rdir[:-14]
"""
root_dir is the root directory of the xaio library.
"""

xaioconfig = root_dir + "/xaio_config.txt"

""" User-defined variable (run configure.py to set it): """

if os.path.exists(xaioconfig):
    with open(xaioconfig, "r") as f:
        lines = f.readlines()

    output_dir = lines[0].rstrip()
    """
    output_dir is the directory in which all outputs should be saved.
    """
