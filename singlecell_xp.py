import numpy as np
from xaio_config import output_dir, xaio_tag
from scRNASeq_preprocessing.load import loadscRNASeq

# from tools.basic_tools import (
#     FeatureTools,
#     confusion_matrix,
#     matthews_coef,
#     umap_plot,
# )

# from tools.feature_selection.RFEExtraTrees import RFEExtraTrees

# from tools.feature_selection.RFENet import RFENet

# from tools.classifiers.LinearSGD import LinearSGD
# import os

from IPython import embed as e

# _ = RFEExtraTrees, RFENet

# data = loadscRNASeq("log")
data = loadscRNASeq("raw")
# data = loadscRNASeq()


data.reduce_features(np.where(data.std_expressions > 0.0)[0])

mitochondrial_genes = data.regex_search(r"\|MT\-")
# data.compute_non_zero_features()
# data.compute_total_counts()


def fun(i):
    # return data.nr_non_zero_features[i]
    # return data.total_counts[i]
    return data.percentage_feature_set(mitochondrial_genes, i)


data.function_plot(fun)

e()

# ft = FeatureTools(data1)

save_dir = output_dir + "/results/scRNASeq/" + xaio_tag + "/"
