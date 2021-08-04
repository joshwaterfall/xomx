from xaio import gdc_create_manifest, gdc_create_data_matrix
from xaio import XAIOData, confusion_matrix, matthews_coef
from xaio import RFEExtraTrees
from xaio import ScoreBasedMulticlass

# from xaio.data_importation.gdc import gdc_create_manifest, gdc_create_data_matrix
# from xaio.tools.basic_tools import XAIOData, confusion_matrix, matthews_coef
# from xaio.tools.feature_selection.RFEExtraTrees import RFEExtraTrees
# from xaio.tools.classifiers.multiclass import ScoreBasedMulticlass
import pandas as pd
import numpy as np
import os
import shutil
from IPython import embed as e

assert e

"""
TUTORIAL: KIDNEY CANCER CLASSIFICATION

The objective of this tutorial is to create and test a classifier of different
types of kidney cancers based on RNA-Seq data from the Cancer Genome Atlas (TCGA).
See kidney_classif.md for detailed explanations.
"""


# The data and outputs will be saved in the following folder:
savedir = os.path.join(os.path.expanduser("~"), "xaiodata", "kidney_classif")
os.makedirs(savedir, exist_ok=True)

"""
STEP 1: Use the gdc_create_manifest function (from xaio/data_importation/gdc.py)
to create a manifest.txt file that will be used to import data with the GDC
Data Transfer Tool (gdc-client). 10 types of cancers are considered, with
for each of them 150 samples corresponding to cases of adenocarcinomas.
"""

if not os.path.exists(os.path.join(savedir, "manifest.txt")):
    # The 3 categories of cancers studied in this tutorial correspond to the following
    # TCGA projects, which are different types of adenocarcinomas:
    disease_type = "Adenomas and Adenocarcinomas"
    project_list = ["TCGA-KIRC", "TCGA-KIRP", "TCGA-KICH"]
    # We will fetch 200 cases of KIRC, 200 cases of KIRP, and 66 cases of KICH
    # from the GDC database:
    case_numbers = [200, 200, 66]
    df_list = gdc_create_manifest(
        disease_type,
        project_list,
        case_numbers,
    )
    df = pd.concat(df_list)
    df.to_csv(
        os.path.join(savedir, "manifest.txt"),
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

# We put the data collected with gdc-client in a temporary folder:
tmpdir = "tmpdir_GDCsamples"
if not os.path.exists(os.path.join(savedir, "xdata")) and (
    not os.path.exists(tmpdir)
    or len(next(os.walk(tmpdir))[1]) <= 0.95 * len(case_numbers)
):
    os.makedirs(tmpdir, exist_ok=True)
    commandstring = (
        "gdc-client download -d "
        + tmpdir
        + " -m "
        + os.path.join(savedir, "manifest.txt")
    )
    os.system(commandstring)
    print("STEP 2: done")
    quit()


"""
STEP 3: Gather all individual cases to create the data matrix, and save it in
a folder named "xdata".
After that, all the individual files imported with gdc-client are erased.
"""

if not os.path.exists(os.path.join(savedir, "xdata")):
    df = gdc_create_data_matrix(
        tmpdir,
        os.path.join(savedir, "manifest.txt"),
    )
    # We drop the last 5 rows containing special information which we will not use:
    df = df.drop(index=df.index[-5:])

    xdata = XAIOData()
    # Import raw data:
    xdata.import_pandas(df)
    # In order to improve cross-sample comparisons, we normalize the sequencing
    # depth to 1 million.
    # WARNING: we use this basic pre-processing to keep the tutorial simple.
    # For more advanced applications, a more sophisticated pre-processing can be
    # required.
    xdata.normalize_feature_sums(1e6)
    # We compute the mean and standard deviation (across samples) for all the features:
    xdata.compute_mean_expressions()
    xdata.compute_std_expressions()
    # Although it will not be used in this tutorial, we compute the number of
    # non-zero features for each sample, and of non-zero samples for each feature:
    xdata.compute_nr_non_zero_features()
    xdata.compute_nr_non_zero_samples()
    xdata.save(["raw"], os.path.join(savedir, "xdata"))
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("STEP 3: done")
    quit()


"""
STEP 4: Annotate the samples.
Annotations are fetched from the previously created file manifest.txt.
"""
if not os.path.exists(
    os.path.join(savedir, "xdata", "sample_indices_per_annotation.npy")
):
    xdata = XAIOData()
    xdata.load(["raw"], os.path.join(savedir, "xdata"))
    manifest = pd.read_table(os.path.join(savedir, "manifest.txt"), header=0)
    xdata.sample_annotations = np.empty(xdata.nr_samples, dtype=object)
    for i in range(xdata.nr_samples):
        xdata.sample_annotations[xdata.sample_indices[manifest["id"][i]]] = manifest[
            "annotation"
        ][i]
    xdata.compute_all_annotations()
    xdata.compute_sample_indices_per_annotation()
    xdata.save(["raw"])
    print("STEP 4: done")
    quit()


"""
STEP 5: Keep only the 4000 features with largest standard deviation, normalize data,
and randomly separate samples in training and test datasets.
"""
if not os.path.exists(os.path.join(savedir, "xdata_small")):
    xdata = XAIOData()
    xdata.load(["raw"], os.path.join(savedir, "xdata"))
    xdata.reduce_features(np.argsort(xdata.std_expressions)[-4000:])
    xdata.compute_normalization("std")
    xdata.compute_train_and_test_indices()
    xdata.compute_std_values_on_training_sets()
    xdata.compute_std_values_on_training_sets_argsort()
    xdata.save(["raw", "std"], os.path.join(savedir, "xdata_small"))
    print("STEP 5: done")
    quit()


"""
STEP 6: Train binary classifiers for every annotation, with recursive feature
elimination to keep 10 features per classifier.
"""
if not os.path.exists(os.path.join(savedir, "xdata_small", "feature_selectors")):
    xdata = XAIOData()
    xdata.load(["raw"], os.path.join(savedir, "xdata_small"))
    nr_annotations = len(xdata.all_annotations)
    feature_selector = np.empty(nr_annotations, dtype=object)
    for i in range(nr_annotations):
        print("Annotation: " + xdata.all_annotations[i])
        feature_selector[i] = RFEExtraTrees(
            xdata,
            xdata.all_annotations[i],
            init_selection_size=4000,
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
    quit()


"""
STEP 7: Visualizing results.
"""
xdata = XAIOData()
xdata.load(["raw"], os.path.join(savedir, "xdata_small"))
xdata.compute_normalization("log")
feature_selector = np.empty(len(xdata.all_annotations), dtype=object)
gene_list = []
for i in range(len(feature_selector)):
    feature_selector[i] = RFEExtraTrees(
        xdata,
        xdata.all_annotations[i],
        init_selection_size=4000,
        n_estimators=450,
        random_state=0,
    )
    feature_selector[i].load(
        os.path.join(
            savedir, "xdata_small", "feature_selectors", xdata.all_annotations[i]
        )
    )
    # feature_selector[i].plot()
    gene_list = gene_list + [
        xdata.feature_names[feature_selector[i].current_feature_indices[j]]
        for j in range(len(feature_selector[i].current_feature_indices))
    ]

# xdata.feature_plot(gene_list, "log")
# The Gene ENSG00000185633.9 (NDUFA4L2) seems associated to KIRC
# (Kidney Renal Clear Cell Carcinoma).
# This is confirmed by the publication:
# Role of NADH Dehydrogenase (Ubiquinone) 1 alpha subcomplex 4-like 2 in clear cell
# renal cell carcinoma
# xdata.feature_plot("ENSG00000185633.9", "raw")


classifier = ScoreBasedMulticlass(xdata.all_annotations, feature_selector)

# all_predictions = classifier.predict(xdata.data_array["raw"][:])
# annot_map = {project_list[i]: i for i in range(len(project_list))}
#
#
# def multi_output(idx):
#     if idx in xdata.test_indices:
#         return annot_map[all_predictions[idx]]
#     else:
#         return -1
#
#
# xdata.function_plot(multi_output, "samples", violinplot_=False)
e()
quit()


"""
STEP 7: Compute the multi-class classifier based on the binary classifiers.
"""


# xdata = XAIOData()
# xdata.load(["raw"], os.path.join(savedir, "xdata_small"))
# xdata.compute_normalization("log")
# feature_selector = np.empty(len(xdata.all_annotations), dtype=object)
# gene_list = []
# for i in range(len(feature_selector)):
#     feature_selector[i] = RFEExtraTrees(
#         xdata, xdata.all_annotations[i], init_selection_size=4000
#     )
#     feature_selector[i].load(os.path.join(savedir, "xdata_small",
#                                           "feature_selectors",
#                                           xdata.all_annotations[i]))
#     feature_selector[i].plot()
#     gene_list = gene_list + [
#             xdata.feature_names[
#                 feature_selector[i].current_feature_indices[j]
#             ]
#             for j in range(len(feature_selector[i].current_feature_indices))
#         ]
# print(gene_list)
# xdata.feature_plot(gene_list, "log")

# # gene: KLK3
# feat_i = xdata.regex_search(r"ENSG00000142515")[0]
# xdata.function_plot(lambda idx: xdata.data[idx, feat_i], "samples")

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
