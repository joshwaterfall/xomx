from data_importation.gdc import gdc_create_manifest
import pandas as pd
import os

"""

TUTORIAL: ADENOCARCINOMA CLASSIFICATION

The objective of this tutorial is to create and test a classifier of different
types of adenocarcinomas based on RNA-seq data from the Cancer Genome Atlas (TCGA).
See adenocarcinomas_classif.md for detailed explanations.

"""


"""

STEP 1: Use the gdc_create_manifest function (from xaio/data_importation/gdc.py)
to create a manifest.txt file that will be used to import data with the GDC
Data Transfer Tool (gdc-client). 10 types of cancers are considered, with
for each of them 150 samples corresponding to cases of adenocarcinomas.

"""

if not os.path.exists("TCGA_samples/manifest.txt"):
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
        r"TCGA_samples/manifest.txt", header=True, index=False, sep="\t", mode="w"
    )
    quit()
