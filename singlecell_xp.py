# import numpy as np
from xaio_config import output_dir, xaio_tag
from tools.basic_tools import RNASeqData

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

data = RNASeqData()
data.save_dir = output_dir + "/dataset/scRNASeq/"
data.load(["raw", "std", "log"])

# data = loadscRNASeq("log")
# data = loadscRNASeq("raw")
# data = loadscRNASeq()

# data.reduce_features(np.where(data.nr_non_zero_samples > 2)[0])
#
mitochondrial_genes = data.regex_search(r"\|MT\-")
# mt_percents = np.array(
#     [
#         data.percentage_feature_set(mitochondrial_genes, i)
#         for i in range(data.nr_samples)
#     ]
# )
#
# data.reduce_samples(np.where(mt_percents < 0.05)[0])
# data.reduce_samples(np.where(data.nr_non_zero_features < 2500)[0])


def tsums(i):
    return data.total_sums[i]


def mt_p(i):
    return data.percentage_feature_set(mitochondrial_genes, i)


def nzfeats(i):
    return data.nr_non_zero_features[i]


def stdval(i):
    return data.std_expressions[i]


def mval(i):
    return data.mean_expressions[i]


data.function_plot(tsums, "samples")

data.function_scatter(tsums, nzfeats, "samples")

data.function_scatter(mval, stdval, "features")

e()

# ft = FeatureTools(data1)

save_dir = output_dir + "/results/scRNASeq/" + xaio_tag + "/"
