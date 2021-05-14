import numpy as np
from xaio_config import output_dir, xaio_tag
from tools.basic_tools import (
    RNASeqData,
    confusion_matrix,
    matthews_coef,
)
from sklearn.cluster import KMeans
from tools.feature_selection.RFEExtraTrees import RFEExtraTrees

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


# data.function_plot(tsums, "samples")
#
# data.function_scatter(tsums, nzfeats, "samples")
#
# data.function_scatter(mval, stdval, "features")

# data.function_plot(lambda i: data.data[i, data.feature_shortnames_ref['MALAT1']],
#                    "samples", violinplot_=False)

data.reduce_features(np.argsort(data.std_expressions)[-4000:])

kmeans = KMeans(n_clusters=8, random_state=0).fit(data.data)
data.sample_annotations = kmeans.labels_
data.compute_all_annotations()
data.compute_sample_indices_per_annotation()
data.compute_train_and_test_indices_per_annotation()
data.compute_std_values_on_training_sets()
data.compute_std_values_on_training_sets_argsort()

for annotation in range(8):
    feature_selector = RFEExtraTrees(data, annotation, init_selection_size=4000)

    print("Initialization...")
    feature_selector.init()
    for siz in [100, 30, 20, 15, 10]:
        print("Selecting", siz, "features...")
        feature_selector.select_features(siz)
        cm = confusion_matrix(
            feature_selector, feature_selector.data_test, feature_selector.target_test
        )
        print("MCC score:", matthews_coef(cm))
    # feature_selector.save(save_dir)
    print("Done.")
    feature_selector.plot()
    # print("MCC score:", matthews_coef(cm))

    print(feature_selector.current_feature_indices)
    print(
        [
            data.feature_names[feature_selector.current_feature_indices[i]]
            for i in range(len(feature_selector.current_feature_indices))
        ]
    )


e()

# ft = FeatureTools(data1)

save_dir = output_dir + "/results/scRNASeq/" + xaio_tag + "/"
