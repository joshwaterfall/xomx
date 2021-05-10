import os

rdir = os.path.abspath(__file__)
assert rdir.endswith("xaio_config.py")
root_dir = rdir[:-14]
"""
root_dir is the root directory of the xaio library.
"""

xaioconfig = root_dir + "/xaio_config.txt"

""" User-defined variables (run configure.py to define them): """

if os.path.exists(xaioconfig):
    with open(xaioconfig, "r") as f:
        lines = f.readlines()

    output_dir = lines[0].rstrip()
    """
    output_dir is the directory where all outputs will be saved.
    """

""" Other config variables: """

xaio_tag = "xaio_tag_1"
