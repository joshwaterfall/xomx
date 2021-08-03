# from tools.sequence_analysis import sequence_to_onehot_vec
from xaio.tools.basic_tools import XAIOData

# from xaio.xaio_config import output_dir
from IPython import embed as e
import pandas as pd
import numpy as np
import os

# import random
from biotransformers import BioTransformers  # pip install bio-transformers

assert BioTransformers
assert e
# from tools.feature_selection.RFEExtraTrees import RFEExtraTrees


def get_reads(filename):
    seq_read = pd.read_table(
        filename,
        sep=",",
        header=0,
        engine="c",
    ).to_numpy()
    trb_indices = np.where(
        # [seq_read[i, 4].startswith("IGKV3D") for i in range(seq_read.shape[0])]
        # [seq_read[i, 4].startswith("IGL") for i in range(seq_read.shape[0])]
        # [seq_read[i, 4].startswith("TRA") for i in range(seq_read.shape[0])]
        # [seq_read[i, 4].startswith("IGK") for i in range(seq_read.shape[0])]
        [seq_read[i_, 4].startswith("IGH") for i_ in range(seq_read.shape[0])]
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


if True:
    dico_3mers = {}
    df = pd.read_csv(
        "protVec_100d_3grams.csv", header=None, sep='\\t|"', engine="python"
    ).to_numpy()
    for i in range(df.shape[0]):
        dico_3mers[df[i][1]] = np.array(df[i][2:102], dtype="float32")

    def protvec(aminoseq):
        mers = [aminoseq[i_ : i_ + 3] for i_ in range(len(aminoseq) - 2)]
        s = np.zeros(100)
        for m in mers:
            if m in dico_3mers:
                s += dico_3mers[m]
            else:
                s += dico_3mers["<unk>"]
        return s


data = XAIOData()
data.save_dir = "/home/perrin/Desktop/data/xaio/dataset/MiXCR/subset_new/"

if not os.path.exists(data.save_dir):
    pass
