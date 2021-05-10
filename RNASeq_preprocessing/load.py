import os
import numpy as np
from tools.basic_tools import RNASeqData
from xaio_config import output_dir


def loadRNASeq(normalization=""):
    data = RNASeqData()
    data.data_dir = os.path.expanduser(output_dir + "/dataset/")
    data.samples_id = np.load(data.data_dir + "samples_id.npy", allow_pickle=True)
    data.annot_dict = np.load(
        data.data_dir + "annot_dict.npy", allow_pickle=True
    ).item()
    data.annot_types = np.load(
        data.data_dir + "annot_types.npy", allow_pickle=True
    ).item()
    data.annot_types_dict = np.load(
        data.data_dir + "annot_types_dict.npy", allow_pickle=True
    ).item()
    data.annot_values = np.load(data.data_dir + "annot_values.npy", allow_pickle=True)
    data.annot_index = np.load(
        data.data_dir + "annot_index.npy", allow_pickle=True
    ).item()
    data.nr_transcripts = np.load(
        data.data_dir + "nr_transcripts.npy", allow_pickle=True
    ).item()
    data.nr_samples = np.load(
        data.data_dir + "nr_samples.npy", allow_pickle=True
    ).item()
    data.transcripts = np.load(data.data_dir + "transcripts.npy", allow_pickle=True)
    data.mean_expressions = np.load(
        data.data_dir + "mean_expressions.npy", allow_pickle=True
    )
    data.std_expressions = np.load(
        data.data_dir + "std_expressions.npy", allow_pickle=True
    )
    data.annot_index_train = np.load(
        data.data_dir + "annot_index_train.npy", allow_pickle=True
    ).item()
    data.annot_index_test = np.load(
        data.data_dir + "annot_index_test.npy", allow_pickle=True
    ).item()
    data.gene_dict = np.load(data.data_dir + "gene_dict.npy", allow_pickle=True).item()
    data.expressions_on_training_sets = np.load(
        data.data_dir + "expressions_on_training_sets.npy", allow_pickle=True
    ).item()
    data.expressions_on_training_sets_argsort = np.load(
        data.data_dir + "expressions_on_training_sets_argsort.npy", allow_pickle=True
    ).item()
    if normalization == "log":
        data.normalization_type = "log"
        data.data = np.array(
            np.memmap(
                data.data_dir + "lognorm_data.bin",
                dtype="float32",
                mode="r",
                shape=(data.nr_transcripts, data.nr_samples),
            )
        ).transpose()
        data.epsilon_shift = np.load(
            data.data_dir + "epsilon_shift.npy", allow_pickle=True
        ).item()
        data.maxlog = np.load(data.data_dir + "maxlog.npy", allow_pickle=True).item()
    elif normalization == "raw":
        data.normalization_type = "raw"
        data.data = np.array(
            np.memmap(
                data.data_dir + "raw_data.bin",
                dtype="float32",
                mode="r",
                shape=(data.nr_transcripts, data.nr_samples),
            )
        ).transpose()
    else:
        data.normalization_type = "mean_std"
        data.data = np.array(
            np.memmap(
                data.data_dir + "data.bin",
                dtype="float32",
                mode="r",
                shape=(data.nr_transcripts, data.nr_samples),
            )
        ).transpose()
    return data
