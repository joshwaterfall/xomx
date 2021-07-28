from xaio.data_importation.gdc import gdc_create_manifest, gdc_create_data_matrix
from xaio.tools.basic_tools import XAIOData
import pandas as pd
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
    not os.path.exists(os.path.join("TCGA_samples", "data_matrix.csv.gz"))
    and len(next(os.walk("TCGA_samples"))[1]) <= 1480
):
    os.system(
        "gdc-client download -d TCGA_samples -m "
        + os.path.join("TCGA_samples", "manifest.txt")
    )
    print("STEP 2: done")
    quit()

"""
STEP 3: Gather all individual cases to create the data matrix, and save it as
data_matrix.csv.gz in the TCGA_samples folder.
After that, all the individual files imported with gdc-client are erased.
"""
if not os.path.exists(os.path.join("TCGA_samples", "data_matrix.csv.gz")):
    gdc_create_data_matrix(
        "TCGA_samples",
        os.path.join("TCGA_samples", "manifest.txt"),
        "data_matrix.csv.gz",
    )
    for sample_dir in next(os.walk("TCGA_samples"))[1]:
        shutil.rmtree(os.path.join("TCGA_samples", sample_dir), ignore_errors=True)
        # os.rmdir(os.path.join("TCGA_samples", sample_dir))

    print("STEP 3: done")
    quit()

xdata = XAIOData()
df = pd.read_table(os.path.join("TCGA_samples", "data_matrix.csv.gz"))
# xdata.import_panda(df)

e()
# manifest = pd.read_table(os.path.join("TCGA_samples", "manifest.txt"))
# df_list = []
# nr_of_samples = manifest.shape[0]
# for i in range(100):
#     if not i % 10:
#         print("  " + str(i) + "/" + str(nr_of_samples), end="\r")
#     if os.path.exists(os.path.join(
#             "TCGA_samples", manifest["id"][i], manifest["filename"][i])
#     ):
#         df_list.append(pd.read_table(
#             os.path.join("TCGA_samples", manifest["id"][i], manifest["filename"][i]),
#             header=None
#         ).rename(columns={1: manifest["id"][i]}).set_index(0))
#
# df_total = df_list[0].join(df_list[1:])
# df_total.index.name = None
# df_total.to_csv(
#     os.path.join("TCGA_samples", "data_matrix.csv"),
#     header=True, index=True, sep="\t", mode="w"
# )
