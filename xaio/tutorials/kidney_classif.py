from xaio import gdc_create_manifest, gdc_create_data_matrix
from xaio import XAIOData, confusion_matrix, matthews_coef
from xaio import RFEExtraTrees

import argparse
import pandas as pd
import numpy as np
import os
import shutil

"""
TUTORIAL: KIDNEY CANCER CLASSIFICATION

The objective of this tutorial is to create and test a classifier of different
types of kidney cancers based on RNA-Seq data from the Cancer Genome Atlas (TCGA).
See kidney_classif.md for detailed explanations.
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step", metavar="S", type=int, nargs="?", default=None, help="execute step S"
    )
    parser.add_argument(
        "--savedir",
        default=os.path.join(
            os.path.expanduser("~"), "results", "xaio", "kidney_classif"
        ),
        help="directory in which data and outputs will be stored",
    )
    args_ = parser.parse_args()
    return args_


# Unless specified otherwise, the data and outputs will be saved in the
# directory: ~/results/xaio/kidney_classif
args = get_args()
savedir = args.savedir
os.makedirs(savedir, exist_ok=True)

# We use the file next_step.txt to know which step to execute next. 7 consecutive
# executions of the code complete the 7 steps of the tutorial.
# A specific step can also be chosen using an integer in argument
# (e.g. `python kidney_classif.py 1` to execute step 1).
os.makedirs(savedir, exist_ok=True)
if args.step is not None:
    assert 1 <= args.step <= 7
    step = args.step
elif not os.path.exists(os.path.join(savedir, "next_step.txt")):
    step = 1
else:
    step = np.loadtxt(os.path.join(savedir, "next_step.txt"), dtype="int")
print("STEP", step)


"""
STEP 1: Use the gdc_create_manifest function (from xaio/data_importation/gdc.py)
to create a manifest.txt file that will be used to import data with the GDC
Data Transfer Tool (gdc-client). 10 types of cancers are considered, with
for each of them 150 samples corresponding to cases of adenocarcinomas.
"""
if step == 1:
    disease_type = "Adenomas and Adenocarcinomas"
    # The 3 categories of cancers studied in this tutorial correspond to the following
    # TCGA projects, which are different types of adenocarcinomas:
    project_list = ["TCGA-KIRC", "TCGA-KIRP", "TCGA-KICH"]
    # We fetch 200 cases of KIRC, 200 cases of KIRP, and 65 cases of KICH from the
    # GDC database:
    case_numbers = [200, 200, 65]
    df_list = gdc_create_manifest(
        disease_type,
        project_list,
        case_numbers,
    )
    df = pd.concat(df_list)
    # noinspection PyTypeChecker
    df.to_csv(
        os.path.join(savedir, "manifest.txt"),
        header=True,
        index=False,
        sep="\t",
        mode="w",
    )
    print("STEP 1: done")


"""
STEP 2: Collect the data with gdc-client (which can be downloaded at
https://gdc.cancer.gov/access-data/gdc-data-transfer-tool).
If all downloads succeed, 465 directories are created in a temporary directory
named tmpdir_GDCsamples.
"""
tmpdir = "tmpdir_GDCsamples"
if step == 2:
    os.makedirs(tmpdir, exist_ok=True)
    commandstring = (
        "gdc-client download -d "
        + tmpdir
        + " -m "
        + os.path.join(savedir, "manifest.txt")
    )
    os.system(commandstring)
    print("STEP 2: done")


"""
STEP 3: Gather all individual cases to create the data matrix, and save it in
a folder named "xdata".
After that, all the individual files imported with gdc-client are erased.
"""
if step == 3:
    df = gdc_create_data_matrix(
        tmpdir,
        os.path.join(savedir, "manifest.txt"),
    )
    # We drop the last 5 rows containing special information which we will not use:
    df = df.drop(index=df.index[-5:])

    xdata = XAIOData()
    # Importing raw data:
    xdata.import_pandas(df)

    # In order to improve cross-sample comparisons, we normalize the sequencing
    # depth to 1 million.
    # WARNING: basic pre-processing is used here for simplicity, but for more advanced
    # applications, a more sophisticated pre-processing may be required.
    xdata.normalize_feature_sums(1e6)

    # We compute the mean and standard deviation (across samples) for all the features:
    xdata.compute_feature_mean_values()
    xdata.compute_feature_standard_deviations()

    # Saving the XAIOData object and its "raw" data array to the disk:
    xdata.save(["raw"], os.path.join(savedir, "xdata"))

    # We erase the individual sample directories downloaded with gdc-client:
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("STEP 3: done")


"""
STEP 4: Annotate the samples.
Annotations are fetched from the previously created file manifest.txt.
"""
if step == 4:
    xdata = XAIOData()
    # Loading the XAIOData object (with normalization_types_list=None, the data array
    # is not loaded):
    xdata.load(normalization_types_list=None, load_dir=os.path.join(savedir, "xdata"))
    manifest = pd.read_table(os.path.join(savedir, "manifest.txt"), header=0)
    xdata.sample_annotations = np.empty(xdata.nr_samples, dtype=object)
    for i in range(xdata.nr_samples):
        xdata.sample_annotations[xdata.sample_indices[manifest["id"][i]]] = manifest[
            "annotation"
        ][i]
    # Computing the list of different annotations:
    xdata.compute_all_annotations()
    # Computing the list of sample indices for every annotation:
    xdata.compute_sample_indices_per_annotation()
    xdata.save()
    print("STEP 4: done")


"""
STEP 5: Keep only the 4000 features with largest standard deviation, normalize data,
and randomly separate samples in training and test datasets.
"""
if step == 5:
    xdata = XAIOData()
    xdata.load(["raw"], os.path.join(savedir, "xdata"))
    xdata.reduce_features(np.argsort(xdata.feature_standard_deviations)[-4000:])
    xdata.compute_train_and_test_indices(test_train_ratio=0.25)
    xdata.save(["raw"], os.path.join(savedir, "xdata_small"))
    print("STEP 5: done")


"""
STEP 6: Train binary classifiers for every annotation, with recursive feature
elimination to keep 10 features per classifier.
"""
if step == 6:
    xdata = XAIOData()
    xdata.load(["raw"], os.path.join(savedir, "xdata_small"))
    nr_annotations = len(xdata.all_annotations)
    feature_selector = np.empty(nr_annotations, dtype=object)
    for i in range(nr_annotations):
        print("Annotation: " + xdata.all_annotations[i])
        feature_selector[i] = RFEExtraTrees(
            xdata,
            xdata.all_annotations[i],
            n_estimators=450,
            random_state=0,
        )
        feature_selector[i].init()
        for siz in [100, 30, 20, 15, 10]:
            print("Selecting", siz, "features...")
            feature_selector[i].select_features(siz)
            cm = confusion_matrix(
                feature_selector[i],
                feature_selector[i].data_test,
                feature_selector[i].target_test,
            )
            print("MCC score:", matthews_coef(cm))
        feature_selector[i].save(
            os.path.join(
                savedir, "xdata_small", "feature_selectors", xdata.all_annotations[i]
            )
        )
        print("Done.")

    print("STEP 6: done")


"""
STEP 7: Visualizing results.
"""
if step == 7:
    xdata = XAIOData()
    xdata.load(["raw"], os.path.join(savedir, "xdata_small"))

    xdata.compute_normalization("std")
    xdata.function_scatter(
        lambda idx: xdata.feature_mean_values[idx],
        lambda idx: xdata.feature_standard_deviations[idx],
        "features",
        xlog_scale=True,
        ylog_scale=True,
    )

    feature_selector = np.empty(len(xdata.all_annotations), dtype=object)
    gene_list = []
    for i in range(len(feature_selector)):
        feature_selector[i] = RFEExtraTrees(
            xdata,
            xdata.all_annotations[i],
            n_estimators=450,
            random_state=0,
        )
        feature_selector[i].load(
            os.path.join(
                savedir, "xdata_small", "feature_selectors", xdata.all_annotations[i]
            )
        )
        gene_list += [
            xdata.feature_names[idx_]
            for idx_ in feature_selector[i].current_feature_indices
        ]

    feature_selector[0].plot()

    xdata.reduce_features(gene_list)
    xdata.compute_normalization("log")
    xdata.umap_plot("log")

    xdata.feature_plot(gene_list, "log")
    xdata.feature_plot("ENSG00000168269.8")  # FOXI1
    xdata.feature_plot("ENSG00000163435.14")  # ELF3
    xdata.feature_plot("ENSG00000185633.9")  # NDUFA4L2

    # Some of the most remarkable genes on this plot:
    # ENSG00000185633.9
    # ENSG00000168269.8 for KICH: FOXI1, known in
    # "Cell-Type-Specific Gene Programs of the Normal Human
    # Nephron Define Kidney Cancer Subtypes"

    # For KIRP: ELF3 ENSG00000163435.14
    # Diagnostic
    # biomarkers
    # for renal cell carcinoma: selection
    # using
    # novel
    # bioinformatics
    # systems
    # for microarray data analysis

    # The Gene ENSG00000185633.9 (NDUFA4L2) seems associated to KIRC
    # (Kidney Renal Clear Cell Carcinoma).
    # This is confirmed by the publication:
    # Role of NADH Dehydrogenase (Ubiquinone) 1 alpha subcomplex 4-like 2 in clear cell
    # renal cell carcinoma
    # xdata.feature_plot("ENSG00000185633.9", "raw")

# noinspection PyTypeChecker
np.savetxt(os.path.join(savedir, "next_step.txt"), [min(step + 1, 7)], fmt="%u")
