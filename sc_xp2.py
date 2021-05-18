# import numpy as np
from xaio_config import output_dir, xaio_tag
from tools.basic_tools import (
    RNASeqData,
    confusion_matrix,
    matthews_coef,
)
from tools.feature_selection.RFEExtraTrees import RFEExtraTrees
from tools.classifiers.multiclass import ScoreBasedMulticlass
from IPython import embed as e

data = RNASeqData()
data.save_dir = output_dir + "/dataset/scRNASeqKMEANS/"
data.load(["raw", "std", "log"])

n_clusters = 8
gene_list = []

classifier = ScoreBasedMulticlass()
classifier.all_annotations = data.all_annotations

if False:
    for annotation in data.all_annotations:
        classifier.binary_classifiers[annotation] = RFEExtraTrees(
            data, annotation, init_selection_size=4000
        )

        print("Initialization...")
        classifier.binary_classifiers[annotation].init()
        for siz in [100, 30, 20, 15, 10]:
            print("Selecting", siz, "features...")
            classifier.binary_classifiers[annotation].select_features(siz)
            cm = confusion_matrix(
                classifier.binary_classifiers[annotation],
                classifier.binary_classifiers[annotation].data_test,
                classifier.binary_classifiers[annotation].target_test,
            )
            print("MCC score:", matthews_coef(cm))
        # feature_selector.save(save_dir)
        print("Done.")
        classifier.binary_classifiers[annotation].plot()
        # print("MCC score:", matthews_coef(cm))

        print(classifier.binary_classifiers[annotation].current_feature_indices)
        gene_list = gene_list + [
            data.feature_names[
                classifier.binary_classifiers[annotation].current_feature_indices[i]
            ].split("|")[1]
            for i in range(
                len(classifier.binary_classifiers[annotation].current_feature_indices)
            )
        ]
        print(gene_list)

    classifier.save(output_dir + "/results/scRNASeq/" + xaio_tag + "/multiclassif/")
else:
    for annotation in data.all_annotations:
        classifier.binary_classifiers[annotation] = RFEExtraTrees(
            data, annotation, init_selection_size=4000
        )

    classifier.load(output_dir + "/results/scRNASeq/" + xaio_tag + "/multiclassif/")

e()
quit()

data.feature_plot(gene_list, "log")

e()
