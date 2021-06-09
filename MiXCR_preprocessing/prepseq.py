# from tools.sequence_analysis import sequence_to_onehot_vec
from tools.basic_tools import RNASeqData
from xaio_config import output_dir
from IPython import embed as e
import pandas as pd
import numpy as np
import random
from biotransformers import BioTransformers

# from tools.feature_selection.RFEExtraTrees import RFEExtraTrees


def get_reads(filename):
    seq_read = pd.read_table(
        filename,
        sep=",",
        header=0,
        engine="c",
    ).to_numpy()
    trb_indices = np.where(
        [seq_read[i, 4].startswith("TRB") for i in range(seq_read.shape[0])]
        # [seq_read[i, 4].startswith("TRA") for i in range(seq_read.shape[0])]
        # [seq_read[i, 4].startswith("IGK") for i in range(seq_read.shape[0])]
    )[0]
    seq_read = np.take(seq_read, trb_indices, axis=0)
    seq_read = np.take(
        seq_read,
        np.where([type(seq_read[i_, 0]) == str for i_ in range(seq_read.shape[0])])[0],
        axis=0,
    )
    seq_read = np.take(seq_read, [0, 9], axis=1)
    return seq_read


def analyze_seq(seq_rd, annotation_, seq_dict=None):
    if seq_dict is None:
        seq_dict = {}
    for i in range(seq_rd.shape[0]):
        # seq_dict.setdefault(seq_rd[i, 1][4:9], {})
        # seq_dict[seq_rd[i, 1][4:9]].setdefault(annotation_, [])
        # if seq_rd[i, 0] not in seq_dict[seq_rd[i, 1][4:9]][annotation_]:
        #     seq_dict[seq_rd[i, 1][4:9]][annotation_].append(seq_rd[i, 0])
        seq_dict.setdefault(seq_rd[i, 1][:], {})
        seq_dict[seq_rd[i, 1][:]].setdefault(annotation_, [])
        if seq_rd[i, 0] not in seq_dict[seq_rd[i, 1][:]][annotation_]:
            seq_dict[seq_rd[i, 1][:]][annotation_].append(seq_rd[i, 0])

    # keys_to_erase = []
    # for key in seq_dict:
    #     totlen = 0
    #     for kk in seq_dict[key]:
    #         totlen += len(seq_dict[key][kk])
    #     if totlen < 2:
    #         keys_to_erase.append(key)
    # for key in keys_to_erase:
    #     del seq_dict[key]


data = RNASeqData()
data.save_dir = output_dir + "/dataset/MiXCR/subset_RECUR/"

if False:
    prefix = (
        "/home/perrin/Desktop/data/"
        + "MiXCR/TCGA_MiXCR/TCGA_MiXCR_NicolasPerrin/Legacy_fileIDs/"
    )
    sdic = {}

    # annotation = "UVM"
    # analyze_seq(get_reads(prefix + "tcga_" + annotation + "_legacy_file_ids.txt"),
    #             annotation, sdic)
    # annotation = "STAD"
    # analyze_seq(get_reads(prefix + "tcga_" + annotation + "_legacy_file_ids.txt"),
    #             annotation, sdic)

    for annotation in [
        "ACC",
        "BLCA",
        "BRCA",
        "CESC",
        "CHOL",
        "COAD",
        "DLBC",
        "ESCA",
        "GBM",
        "HNSC",
        "KICH",
        "KIRC",
        "KIRP",
        "LGG",
        "LIHC",
        "LUAD",
        "LUSC",
        "MESO",
        "OV",
        "PAAD",
        "PCPG",
        "PRAD",
        "READ",
        "SARC",
        "SKCM",
        "STAD",
        "TGCT",
        "THCA",
        "THYM",
        "UCEC",
        "UCS",
        "UVM",
    ]:
        # for annotation in ["BRCA", "PAAD", "STAD", "BLCA", "CHOL",
        #                    "THYM", "LUAD", "ESCA", "DLBC"]:
        # for annotation in ["PAAD", "STAD"]:
        # for annotation in ["ACC", "BLCA"]:
        print(annotation)
        analyze_seq(
            get_reads(prefix + "tcga_" + annotation + "_legacy_file_ids.txt"),
            annotation,
            sdic,
        )

    if True:
        keys_to_erase = []
        for key in sdic:
            totlen = 0
            for kk in sdic[key]:
                totlen += len(sdic[key][kk])
            # if totlen > 1 or len(key) < 20:
            if totlen < 2:
                # if len(key) != 19:
                keys_to_erase.append(key)
        for key in keys_to_erase:
            del sdic[key]

    # for key in sdic:
    #     sdic[key]

    data.sample_ids = np.array(list(sdic.keys()))
    # data.sample_ids = np.empty(len(sdic), dtype=object)
    # for i, key in enumerate(sdic.keys()):
    #     data.sample_ids[i] = key[4:8]

    data.sample_annotations = np.empty_like(data.sample_ids)
    for i, s_id in enumerate(data.sample_ids):
        data.sample_annotations[i] = "_".join(list(sdic[s_id].keys()))
    data.compute_sample_indices()
    data.compute_all_annotations()
    data.compute_sample_indices_per_annotation()

    keep_samples = []
    e()
    quit()
    # for annot in data.all_annotations:
    for annot in [
        "ACC",
        "BLCA",
        "BRCA",
        "CESC",
        "CHOL",
        "DLBC",
        "ESCA",
        "HNSC",
        "KIRC",
        "KIRP",
        "LIHC",
        "LUAD",
        "LUSC",
        "MESO",
        "OV",
        "PAAD",
        "PRAD",
        "SARC",
        "SKCM",
        "STAD",
        "TGCT",
        "THCA",
        "THYM",
        "UCEC",
    ]:
        nr_s = min(len(data.sample_indices_per_annotation[annot]), 150)
        keep_samples += random.sample(data.sample_indices_per_annotation[annot], nr_s)

    data.reduce_samples(keep_samples)
    data.save([])

# e()
# quit()
data.load([])

if False:
    sequences = list(data.sample_ids)
    # sequences = [
    #         "CASSYSGTDEQYF",
    #         "CASSPSTDTQYF",
    #     ]

    # bio_trans = BioTransformers(backend="protbert")
    bio_trans = BioTransformers(backend="esm1b_t33_650M_UR50S")
    embeddings = bio_trans.compute_embeddings(sequences, pool_mode=("cls", "mean"))

    cls_emb = embeddings["cls"]
    mean_emb = embeddings["mean"]

    # maxlength = 20
    # data.nr_features = maxlength * 21

    data.nr_features = 1280
    # data.nr_features = 1024

    data.nr_samples = len(data.sample_ids)
    print("nr of samples:", data.nr_samples)

    data.data_array["raw"] = cls_emb

    # data.data_array["raw"] = np.zeros((data.nr_samples, data.nr_features))
    # for i, s_id in enumerate(data.sample_ids):
    #     data.data_array["raw"][i, :] = sequence_to_onehot_vec(s_id, maxlength)

    # data.feature_names = np.empty((data.nr_features,), dtype=object)
    # mapping = "ARNDCQEGHILKMFPSTWYVX"
    # for i in range(maxlength):
    #     for j in range(21):
    #         data.feature_names[i * 21 + j] = "|" + mapping[j] + str(i)

    data.save(["raw"])

# e()
# quit()

data.load(["raw"])
# data.reduce_samples(data.sample_indices_per_annotation["LUSC"]
#                     + data.sample_indices_per_annotation["STAD"])
# data.compute_train_and_test_indices_per_annotation()
# feature_selector = RFEExtraTrees(data, "STAD", init_selection_size=1280)
# feature_selector.init()
# feature_selector.select_features(1280)

e()

data.umap_plot("raw")

e()
# "-".join(list(sdic["CASSPGSYEQYF"].keys()))
