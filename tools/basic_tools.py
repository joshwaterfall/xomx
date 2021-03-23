# import os
import numpy as np

# import matplotlib.pyplot as plt
# import string
# import random
from xaio_config import data_dir


class RNASeqData:
    """
    Attributes:

    samples_id -> array of IDs: i-th sample has ID samples_id[i] (starting at 0)

    annot_dict -> dict of annotations: sample of ID "#" has annotation annot_dict["#"]

    annot_values -> list of all annotations

    annot_index -> annot_index["#"] is the list of indices of the samples of
                   annotation "#"

    nr_transcripts -> the total number of features (i.e. transcripts) for each sample

    nr_samples -> the total number of samples

    transcripts -> IDs for the features: i-th feature has ID transcripts[i]

    mean_expressions -> mean_expression[i] is the mean value of the i-th feature
                            accross all samples

    std_expressions -> similar to mean_expression but standard deviation instead of mean

    annot_index_train -> annot_index["#"] is the list of indices of the samples of
                         annotation "#" which belong to the training set

    annot_index_test -> annot_index["#"] is the list of indices of the samples of
                        annotation "#" which belong to the validation set

    gene_dict -> features have short IDs: gene_dict["#"] is the index of the feature
                 of short ID #

    expressions_on_training_sets -> expressions_on_training_sets["#"][j] is the mean
                                    value of the j-th feature in the data normalized
                                    by mean and std_dev for all samples of annotation
                                    "#" belonging to the training set

    expressions_on_training_sets_argsort -> expressions_on_training_sets_argsort["#"]
                                            is the list of feature indices sorted by
                                            devreasing value in
                                            expressions_on_training_sets["#"]

    annot_types -> annot_types["#"] is a string characterizing the data origin for
                   the sample of ID "#"

    annot_types_dict -> annot_types_dict["#"] is the set of different origins for all
                        the samples of annotation "#"

    epsilon_shift -> the value of the shift used for log-normalization of the data

    maxlog -> maximum value of the log data

    normalization_type -> the type of normalization:
                          - None or "" = ( . - mean) / std_dev
                          - "log" = log_normalization

    data -> data[i, j]: value of the j-th feature of the i-th sample
            if the normalization is of type ( . - mean) / std_dev:
              data[i, j] * std_expression[j] + mean_expression[j] is the original value
            if the normalization is of type log-norm:
              the original value is:
              np.exp( data[i,j] * (maxlog - np.log(epsilon_shift))
                     + np.log(epsilon_shift)) - epsilon_shift
    """

    def __init__(self):
        self.data_dir = None
        self.samples_id = None
        self.annot_dict = None
        self.annot_types = None
        self.annot_types_dict = None
        self.annot_values = None
        self.annot_index = None
        self.nr_transcripts = None
        self.nr_samples = None
        self.transcripts = None
        self.mean_expressions = None
        self.std_expressions = None
        self.data = None
        self.annot_index_train = None
        self.annot_index_test = None
        self.gene_dict = None
        self.expressions_on_training_sets = None
        self.expressions_on_training_sets_argsort = None
        self.maxlog = None
        self.epsilon_shift = None
        self.normalization_type = None


def load(normalization=""):
    data = RNASeqData()
    data.samples_id = np.load(data_dir + "samples_id.npy", allow_pickle=True)
    data.annot_dict = np.load(data_dir + "annot_dict.npy", allow_pickle=True).item()
    data.annot_types = np.load(data_dir + "annot_types.npy", allow_pickle=True).item()
    data.annot_types_dict = np.load(
        data_dir + "annot_types_dict.npy", allow_pickle=True
    ).item()
    data.annot_values = np.load(data_dir + "annot_values.npy", allow_pickle=True)
    data.annot_index = np.load(data_dir + "annot_index.npy", allow_pickle=True).item()
    data.nr_transcripts = np.load(
        data_dir + "nr_transcripts.npy", allow_pickle=True
    ).item()
    data.nr_samples = np.load(data_dir + "nr_samples.npy", allow_pickle=True).item()
    data.transcripts = np.load(data_dir + "transcripts.npy", allow_pickle=True)
    data.mean_expressions = np.load(
        data_dir + "mean_expressions.npy", allow_pickle=True
    )
    data.std_expressions = np.load(data_dir + "std_expressions.npy", allow_pickle=True)
    data.annot_index_train = np.load(
        data_dir + "annot_index_train.npy", allow_pickle=True
    ).item()
    data.annot_index_test = np.load(
        data_dir + "annot_index_test.npy", allow_pickle=True
    ).item()
    data.gene_dict = np.load(data_dir + "gene_dict.npy", allow_pickle=True).item()
    data.expressions_on_training_sets = np.load(
        data_dir + "expressions_on_training_sets.npy", allow_pickle=True
    ).item()
    data.expressions_on_training_sets_argsort = np.load(
        data_dir + "expressions_on_training_sets_argsort.npy", allow_pickle=True
    ).item()
    if normalization != "log":
        data.data = np.array(
            np.memmap(
                data_dir + "data.bin",
                dtype="float32",
                mode="r",
                shape=(data.nr_transcripts, data.nr_samples),
            )
        ).transpose()
    else:
        data.data = np.array(
            np.memmap(
                data_dir + "lognorm_data.bin",
                dtype="float32",
                mode="r",
                shape=(data.nr_transcripts, data.nr_samples),
            )
        ).transpose()
        data.epsilon_shift = np.load(
            data_dir + "epsilon_shift.npy", allow_pickle=True
        ).item()
        data.maxlog = np.load(data_dir + "maxlog.npy", allow_pickle=True).item()
    return data
