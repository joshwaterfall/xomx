# from tools.sequence_analysis import sequence_to_onehot
from IPython import embed as e
import pandas as pd
import numpy as np

seq_read = pd.read_table(
    "/home/perrin/Desktop/data/"
    + "MiXCR/TCGA_MiXCR/TCGA_MiXCR_NicolasPerrin/Legacy_fileIDs/"
    +
    # "tcga_ACC_legacy_file_ids.txt", sep=",", header=0, engine="c").to_numpy()
    # "tcga_BLCA_legacy_file_ids.txt", sep=",", header=0, engine="c").to_numpy()
    # "tcga_LUSC_legacy_file_ids.txt", sep=",", header=0, engine="c").to_numpy()
    # "tcga_LUAD_legacy_file_ids.txt", sep=",", header=0, engine="c").to_numpy()
    # "tcga_STAD_legacy_file_ids.txt", sep=",", header=0, engine="c").to_numpy()
    # "tcga_THCA_legacy_file_ids.txt", sep=",", header=0, engine="c").to_numpy()
    "tcga_OV_legacy_file_ids.txt",
    sep=",",
    header=0,
    engine="c",
).to_numpy()

TRB_indices = np.where(
    [seq_read[i, 4].startswith("TRB") for i in range(seq_read.shape[0])]
)[0]

seq_read = np.take(seq_read, TRB_indices, axis=0)
seq_read = np.take(
    seq_read,
    np.where([type(seq_read[i, 0]) == str for i in range(seq_read.shape[0])])[0],
    axis=0,
)
seq_read = np.take(seq_read, [0, 9], axis=1)

seq_dict = {}
for i in range(seq_read.shape[0]):
    seq_dict.setdefault(seq_read[i, 0], [])
    seq_dict[seq_read[i, 0]].append(seq_read[i, 1])

m = 0.0
for k in seq_dict:
    m = m + len(seq_dict[k])
m = m / len(seq_dict.keys())
e()
