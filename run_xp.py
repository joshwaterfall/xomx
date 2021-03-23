from xaio_config import output_dir, xaio_tag
from tools.basic_tools import load, FeatureTools
from tools.feature_selection.RFEExtraTrees import RFEExtraTrees
import os

# import subprocess
# import numpy as np
from IPython import embed as e

data = load("log")
# data = load()
gt = FeatureTools(data)

annotation = "Breast"

save_dir = os.path.expanduser(
    output_dir + "/results/" + xaio_tag + "/" + annotation.replace(" ", "_") + "/"
)

rfeet = RFEExtraTrees(data, annotation)
rfeet.select_features(3000)

e()
