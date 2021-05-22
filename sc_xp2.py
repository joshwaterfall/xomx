import numpy as np
from xaio_config import output_dir, xaio_tag
from tools.basic_tools import (
    RNASeqData,
    confusion_matrix,
    matthews_coef,
)
from tools.feature_selection.RFEExtraTrees import RFEExtraTrees
from tools.classifiers.multiclass import ScoreBasedMulticlass
from tools.normalization.sctransform import compute_sctransform
from IPython import embed as e

data = RNASeqData()
data.save_dir = output_dir + "/dataset/scRNASeqKMEANS/"
data.load(["raw", "std", "sct"])

if False:
    compute_sctransform(data)
    data.save(["sct"])

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

all_predictions = classifier.predict(data.data)

gene_list = list(
    np.concatenate(
        [
            classifier.binary_classifiers[annot_].current_feature_indices
            for annot_ in [4, 0, 3, 5, 1, 2]
        ]
    )
)

e()
quit()

data.function_plot(lambda i: data.total_sums[i], "samples")

data.function_plot(lambda i: data.nr_non_zero_features[i], "samples")

data.function_scatter(
    lambda i: data.total_sums[i], lambda i: data.nr_non_zero_features[i], "samples"
)

data.function_scatter(
    lambda i: data.mean_expressions[i], lambda i: data.std_expressions[i], "features"
)

classifier.binary_classifiers[0].plot()

data.function_plot(lambda i: all_predictions[i], "samples", 2, violinplot_=False)

data.umap_plot("log")

data.feature_plot(gene_list, "log")

data.feature_plot(["IL7R", "CCR7"], "log")

data.feature_plot(["LYZ", "CD14"], "log")

data.feature_plot(["IL7R", "S100A4"], "log")

data.feature_plot(["MS4A1"], "log")

data.feature_plot(["CD8A"], "log")

data.feature_plot(["FCGR3A", "MS4A7"], "log")

data.feature_plot(["GNLY", "NKG7"], "log")

data.feature_plot(["FCER1A", "CST3"], "log")

data.feature_plot(["PPBP"], "log")


e()
