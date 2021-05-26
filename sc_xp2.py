import numpy as np
from xaio_config import output_dir, xaio_tag
from tools.basic_tools import (
    RNASeqData,
    confusion_matrix,
    matthews_coef,
)
from sklearn.cluster import KMeans
from tools.feature_selection.RFEExtraTrees import RFEExtraTrees
from tools.classifiers.multiclass import ScoreBasedMulticlass
from tools.normalization.sctransform import compute_sctransform

# compute_logsctransform
from IPython import embed as e

data = RNASeqData()
data.save_dir = output_dir + "/dataset/scRNASeqKMEANS/"
data.load(["logsct", "std", "raw"])

# compute_logsctransform(data)

if False:
    compute_sctransform(data)
    data.save(["sct"])

n_clusters = 9
gene_list = []

if False:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(
        data.data_array["logsct"]
    )
    data.sample_annotations = kmeans.labels_
    data.compute_all_annotations()
    data.compute_sample_indices_per_annotation()
    data.compute_train_and_test_indices_per_annotation()
    data.compute_std_values_on_training_sets()
    data.compute_std_values_on_training_sets_argsort()

    data.save(output_dir + "/dataset/scRNASeqKMEANS/")

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

# data.sample_annotations = all_predictions
# data.compute_sample_indices_per_annotation()
# data.compute_train_and_test_indices_per_annotation()
# data.compute_std_values_on_training_sets()
# data.compute_std_values_on_training_sets_argsort()


gene_list = list(
    np.concatenate(
        [
            classifier.binary_classifiers[annot_].current_feature_indices
            # for annot_ in [4, 2, 7, 3, 5 ,1 ,6 ,0]
            for annot_ in range(n_clusters)
        ]
    )
)
# data.reduce_features(gene_list)

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

data.function_plot(lambda i: all_predictions[i], "samples", violinplot_=False)

data.umap_plot("log")

data.feature_plot(gene_list, "logsct")

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
