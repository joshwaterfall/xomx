# from xaio import gdc_create_manifest, gdc_create_data_matrix
from xaio import XAIOData, confusion_matrix, matthews_coef

# from xaio import RFEExtraTrees
# import numpy as np

# from xaio.tools.basic_tools import (
#     XAIOData,
#     confusion_matrix,
#     matthews_coef,
# )
from sklearn.cluster import KMeans
from xaio.tools.feature_selection.RFEExtraTrees import RFEExtraTrees
import scipy.io
import csv
import argparse

# import pandas as pd
import numpy as np
import os
import requests
import tarfile

# import shutil
from IPython import embed as e

assert e

"""
TUTORIAL: PERIPHERAL BLOOD MONONUCLEAR CELLS ANALYSIS

This tutorial is similar to the following tutorial with the R package Seurat:
https://satijalab.org/seurat/articles/pbmc3k_tutorial.html

The objective is to analyze a dataset of Peripheral Blood Mononuclear Cells (PBMC)
freely available from 10X Genomics, composed of 2,700 single cells that were
sequenced on the Illumina NextSeq 500.
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step", metavar="S", type=int, nargs="?", default=None, help="execute step S"
    )
    parser.add_argument(
        "--savedir",
        default=os.path.join(os.path.expanduser("~"), "results", "xaio", "pbmc"),
        help="directory in which data and outputs will be stored",
    )
    args_ = parser.parse_args()
    return args_


# Unless specified otherwise, the data and outputs will be saved in the
# directory: ~/results/xaio/pbmc
args = get_args()
savedir = args.savedir
os.makedirs(savedir, exist_ok=True)

# We use the file next_step.txt to know which step to execute next. 7 consecutive
# executions of the code complete the 7 steps of the tutorial.
# A specific step can also be chosen using an integer in argument
# (e.g. `python kidney_classif.py 1` to execute step 1).
if args.step is not None:
    assert 1 <= args.step <= 7
    step = args.step
elif not os.path.exists(os.path.join(savedir, "next_step.txt")):
    step = 1
else:
    step = np.loadtxt(os.path.join(savedir, "next_step.txt"), dtype="int")
print("STEP", step)

"""
STEP 1: Load data from the 10X Genomics website, and save it as a XAIOData object.
"""
if step == 1:
    url = (
        "https://cf.10xgenomics.com/samples/cell/pbmc3k/"
        + "pbmc3k_filtered_gene_bc_matrices.tar.gz"
    )
    r = requests.get(url, allow_redirects=True)
    open(os.path.join(savedir, "pbmc3k.tar.gz"), "wb").write(r.content)
    tar = tarfile.open(os.path.join(savedir, "pbmc3k.tar.gz"), "r:gz")
    downloaded_content = {}
    for member in tar.getmembers():
        file = tar.extractfile(member)
        if file is not None:
            downloaded_content[member.name] = file

    # The data matrix is loaded as a SciPy sparse matrix.
    sparse_data_matrix = scipy.io.mmread(
        downloaded_content["filtered_gene_bc_matrices/hg19/matrix.mtx"]
    )

    # List of feature names (genes):
    featurenames = [
        "|".join(row)
        for row in list(
            csv.reader(
                downloaded_content["filtered_gene_bc_matrices/hg19/genes.tsv"]
                .read()
                .decode("utf-8")
                .splitlines(),
                delimiter="\t",
            )
        )
    ]
    # List of the sample IDs (barcodes):
    sampleids = list(
        csv.reader(
            downloaded_content["filtered_gene_bc_matrices/hg19/barcodes.tsv"]
            .read()
            .decode("utf-8")
            .splitlines(),
            delimiter="\t",
        )
    )

    # Creation of the XAIOData object:
    xd = XAIOData()
    xd.save_dir = savedir

    # Sparse matrices are not supported yet in XAIO, so we convert the data matrix into
    # a dense matrix before copying it to the XAIOData object. It is transposed because
    # XAIO uses the convention: columns=samples and rows=features.
    xd.data_array["raw"] = sparse_data_matrix.todense().transpose()
    xd.feature_names = featurenames
    xd.sample_ids = sampleids
    xd.save(["raw"])

"""
STEP 2
"""
if step == 2:
    xd = XAIOData()
    xd.save_dir = savedir
    xd.load(["raw"])
    e()
    quit()

data = XAIOData()
# data.save_dir = output_dir + "/dataset/scRNASeq/"
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
    return data.total_feature_sums[i]


def mt_p(i):
    return data.feature_values_ratio(mitochondrial_genes, i)


def nzfeats(i):
    return data.nr_non_zero_features[i]


def stdval(i):
    return data.feature_standard_deviations[i]


def mval(i):
    return data.feature_mean_values[i]


# data.function_plot(tsums, "samples")
#
# data.function_scatter(tsums, nzfeats, "samples")
#
# data.function_scatter(mval, stdval, "features")

# data.function_plot(lambda i: data.data[i, data.feature_shortnames_ref['MALAT1']],
#                    "samples", violinplot_=False)

data.reduce_features(np.argsort(data.feature_standard_deviations)[-4000:])

n_clusters = 6

kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(data.data_array["log"])
data.sample_annotations = kmeans.labels_
data.compute_all_annotations()
data.compute_sample_indices_per_annotation()
data.compute_train_and_test_indices_per_annotation()
data.compute_std_values_on_training_sets()
data.compute_std_values_on_training_sets_argsort()

# data.save_dir = output_dir + "/dataset/scRNASeqKMEANS/"
data.save()

e()
quit()

# list_genes = ["FTL", "LYZ"]
#
#
# def feature_func(x, annot):
#     return data.log_data[x, data.feature_shortnames_ref[annot]]
#
#
# fun_list = [lambda x: feature_func(x, 0), lambda x: feature_func(x, 1)]

# e()
# quit()

gene_list = []

feature_selector = np.empty(n_clusters, dtype=object)

for annotation in range(n_clusters):
    feature_selector[annotation] = RFEExtraTrees(
        data, annotation, init_selection_size=4000
    )

    print("Initialization...")
    feature_selector[annotation].init()
    for siz in [100, 30, 20, 15, 10]:
        print("Selecting", siz, "features...")
        feature_selector[annotation].select_features(siz)
        cm = confusion_matrix(
            feature_selector[annotation],
            feature_selector[annotation].data_test,
            feature_selector[annotation].target_test,
        )
        print("MCC score:", matthews_coef(cm))
    # feature_selector.save(save_dir)
    print("Done.")
    feature_selector[annotation].plot()
    # print("MCC score:", matthews_coef(cm))

    print(feature_selector[annotation].current_feature_indices)
    gene_list = gene_list + [
        data.feature_names[
            feature_selector[annotation].current_feature_indices[i]
        ].split("|")[1]
        for i in range(len(feature_selector[annotation].current_feature_indices))
    ]
    print(gene_list)

e()

data.feature_plot(gene_list, "log")

e()

# ft = FeatureTools(data1)

"""
INCREMENTING next_step.txt
"""
# noinspection PyTypeChecker
np.savetxt(os.path.join(savedir, "next_step.txt"), [min(step + 1, 7)], fmt="%u")
