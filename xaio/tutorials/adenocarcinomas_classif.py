from xaio.data_importation.gdc import gdc_create_manifest, gdc_create_data_matrix
from xaio.tools.basic_tools import XAIOData
import pandas as pd
import numpy as np
import os
import shutil
from IPython import embed as e

assert e

"""
TUTORIAL: ADENOCARCINOMA CLASSIFICATION

The objective of this tutorial is to create and test a classifier of different
types of adenocarcinomas based on RNA-Seq data from the Cancer Genome Atlas (TCGA).
See adenocarcinomas_classif.md for detailed explanations.
"""


"""
STEP 1: Use the gdc_create_manifest function (from xaio/data_importation/gdc.py)
to create a manifest.txt file that will be used to import data with the GDC
Data Transfer Tool (gdc-client). 10 types of cancers are considered, with
for each of them 150 samples corresponding to cases of adenocarcinomas.
"""

if not os.path.exists(os.path.join("TCGA_samples", "manifest.txt")):
    df_list = gdc_create_manifest(
        "Adenomas and Adenocarcinomas",
        [
            "TCGA-KIRC",
            "TCGA-THCA",
            "TCGA-PRAD",
            "TCGA-LUAD",
            "TCGA-UCEC",
            "TCGA-COAD",
            "TCGA-LIHC",
            "TCGA-STAD",
            "TCGA-KIRP",
            "TCGA-READ",
        ],
        [150] * 10,
    )
    df = pd.concat(df_list)
    df.to_csv(
        os.path.join("TCGA_samples", "manifest.txt"),
        header=True,
        index=False,
        sep="\t",
        mode="w",
    )
    print("STEP 1: done")
    quit()


"""
STEP 2: Collect the data with gdc-client (which can be downloaded at
https://gdc.cancer.gov/access-data/gdc-data-transfer-tool).
If all downloads succeed, it creates 1500 subfolders in the TCGA_samples directory.
"""

if (
    not os.path.exists(os.path.join("TCGA_samples", "XAIOData"))
    and len(next(os.walk("TCGA_samples"))[1]) <= 1480
):
    os.system(
        "gdc-client download -d TCGA_samples -m "
        + os.path.join("TCGA_samples", "manifest.txt")
    )
    print("STEP 2: done")
    quit()


"""
STEP 3: Gather all individual cases to create the data matrix, and save it in
the folder TCGA_samples/XAIOData.
After that, all the individual files imported with gdc-client are erased.
"""

if not os.path.exists(os.path.join("TCGA_samples", "XAIOData")):
    df = gdc_create_data_matrix(
        "TCGA_samples",
        os.path.join("TCGA_samples", "manifest.txt"),
    )
    # We drop the last 5 rows that contain special information which we will not use:
    df = df.drop(index=df.index[-5:])

    xdata = XAIOData()
    xdata.import_pandas(df)
    xdata.save(["raw"], os.path.join("TCGA_samples", "XAIOData"))
    for sample_dir in next(os.walk("TCGA_samples"))[1]:
        if sample_dir != "XAIOData":
            shutil.rmtree(os.path.join("TCGA_samples", sample_dir), ignore_errors=True)
    print("STEP 3: done")
    quit()


"""
STEP 4: Annotate the samples and remove features with standard deviation less than 1e-8.
"""
if not os.path.exists(
    os.path.join("TCGA_samples", "XAIOData", "sample_indices_per_annotation.npy")
):
    xdata = XAIOData()
    xdata.load(["raw"], os.path.join("TCGA_samples", "XAIOData"))
    # 1) Fetching annotations from manifest.txt
    manifest = pd.read_table(os.path.join("TCGA_samples", "manifest.txt"), header=0)
    xdata.sample_annotations = np.empty(xdata.nr_samples, dtype=object)
    for i in range(xdata.nr_samples):
        xdata.sample_annotations[xdata.sample_indices[manifest["id"][i]]] = manifest[
            "annotation"
        ][i]
    xdata.compute_all_annotations()
    xdata.compute_sample_indices_per_annotation()
    # 2) Removing barely varying features
    keep_indices = []
    for i in range(xdata.nr_features):
        if xdata.std_expressions[i] > 1e-8:
            keep_indices.append(i)
    xdata.reduce_features(keep_indices)
    xdata.save(["raw"])
    print("STEP 4: done")
    quit()


"""
STEP 5: First visualizations.
"""

xdata = XAIOData()
xdata.load(["raw"], os.path.join("TCGA_samples", "XAIOData"))
xdata.compute_normalization("std")
xdata.compute_train_and_test_indices_per_annotation()
xdata.compute_std_values_on_training_sets()
xdata.compute_std_values_on_training_sets_argsort()
xdata.save(["raw", "std"])
# # 1) Plotting the total sum of counts for each sample:
# xdata.function_plot(lambda idx: xdata.total_sums[idx], "samples")
#
# # 2) Plotting mean value vs. std deviation for all features:
# xdata.function_scatter(lambda idx: xdata.mean_expressions[idx],
#                        lambda idx: xdata.std_expressions[idx],
#                        "features")

# The gene UBE2Q1 is known as a potential biomarker for Lung Adenocarcinoma (LUAD).
# Its Gene ID is ENSG00000160714. We can look for its feature index in our data
# with a regex search:
# ube2q1_index = xdata.regex_search(r"ENSG00000160714")[0]
# ube2q1_index = xdata.regex_search(r"ENSG00000232931")[0]
# ube2q1_index = xdata.regex_search(r"ENSG00000237424")[0]
# ube2q1_index = xdata.regex_search(r"ENSG00000164932")[0]
# assert ube2q1_index == 10516
# Now we plot the value of this feature accross all samples:
# xdata.function_plot(lambda idx: xdata.data[idx, ube2q1_index], "samples")

e()
quit()
