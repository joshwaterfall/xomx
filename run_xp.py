from xaio_config import output_dir, xaio_tag
from tools.basic_tools import load, FeatureTools, confusion_matrix, matthews_coef
from tools.feature_selection.RFEExtraTrees import RFEExtraTrees

# from tools.feature_selection.RFENet import RFENet
from tools.classifiers.LinearSGD import LinearSGD
import os
from IPython import embed as e

# _ = RFEExtraTrees, RFENet

# data = load("log")
data = load()
gt = FeatureTools(data)

annotation = "Brain lower grade glioma"
# annotation = "TCGA-LGG_Primary Tumor"
# annotation = "Breast invasive carcinoma"

save_dir = os.path.expanduser(
    output_dir + "/results/" + xaio_tag + "/" + annotation.replace(" ", "_")
)

e()

# feature_selector = RFENet(data, annotation, init_selection_size=4000)
feature_selector = RFEExtraTrees(data, annotation, init_selection_size=4000)
if not feature_selector.load(save_dir):
    print("Initialization...")
    feature_selector.init()
    for siz in [100, 30, 15, 10]:
        print("Selecting", siz, "features...")
        feature_selector.select_features(siz)
    feature_selector.save(save_dir)
print("Done.")
cm = confusion_matrix(
    feature_selector, feature_selector.data_test, feature_selector.target_test
)
print("MCC score:", matthews_coef(cm))

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
# gene_list = [3301, 47704, 6692, 54294, 583, 16343, 20741, 23090, 34358, 33734]

linear_clf = LinearSGD(data)

linear_clf.fit(feature_selector.data_train, feature_selector.target_train)
linear_clf.plot(
    feature_selector.data_test,
    feature_selector.test_indices,
    annotation,
    save_dir + "/d1/",
)

linear_clf.fit_list(gene_list, annotation)
linear_clf.plot_list(gene_list, None, save_dir + "/d2/")

e()
