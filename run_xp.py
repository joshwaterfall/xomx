from xaio_config import output_dir, xaio_tag
from tools.basic_tools import load, FeatureTools, confusion_matrix
from tools.feature_selection.RFEExtraTrees import RFEExtraTrees
import os

# import subprocess
# import numpy as np
from IPython import embed as e

data = load("log")
# data = load()
gt = FeatureTools(data)

e()
quit()

annotation = "Breast"

save_dir = os.path.expanduser(
    output_dir + "/results/" + xaio_tag + "/" + annotation.replace(" ", "_")
)

rfeet = RFEExtraTrees(data, annotation, init_selection_size=50)
# rfeet.select_features(10)

rfeet.load(save_dir)
print(confusion_matrix(rfeet, rfeet.data_test, rfeet.target_test))


e()
