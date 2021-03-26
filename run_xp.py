from xaio_config import output_dir, xaio_tag
from tools.basic_tools import load, FeatureTools
from tools.feature_selection.RFEExtraTrees import RFEExtraTrees
from tools.classifiers.LinearSGD import LinearSGD
import os

# data = load("log")
data = load()
gt = FeatureTools(data)

annotation = "Brain lower grade glioma"

save_dir = os.path.expanduser(
    output_dir + "/results/" + xaio_tag + "/" + annotation.replace(" ", "_")
)

rfeet = RFEExtraTrees(data, annotation, init_selection_size=4000)
if not rfeet.load(save_dir):
    for siz in [100, 30, 15, 10]:
        print("Selecting", siz, "features.")
        rfeet.select_features(siz)
    rfeet.save(save_dir)

linear_clf = LinearSGD(data)
linear_clf.fit(rfeet.data_train, rfeet.target_train)
linear_clf.plot(rfeet.data_test, rfeet.test_indices, annotation)

gene_list = [
    "PTPRZ1",
    "RP11-977G19.5",
    "BCAN",
    "MIR497HG",
    "MYO16",
    "RP4-592A1.2",
    "RPSAP58",
    "HNRNPA3P6",
    "HMGN2P5",
    "RP11-535M15.2",
]

linear_clf.fit_list(gene_list, annotation)
linear_clf.plot_list(gene_list, annotation)
