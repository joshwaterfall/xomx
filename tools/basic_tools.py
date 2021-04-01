import os
import numpy as np
import matplotlib.pyplot as plt
from xaio_config import output_dir


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
                                    "#" belonging to the training set ; it is useful
                                    to determine whether a transcript is up-regulated
                                    for a given diagnosis (positive value), or
                                    down-regulated (negative value)

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
    if normalization != "log":
        data.data = np.array(
            np.memmap(
                data.data_dir + "data.bin",
                dtype="float32",
                mode="r",
                shape=(data.nr_transcripts, data.nr_samples),
            )
        ).transpose()
    else:
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
    return data


class FeatureTools:
    def __init__(self, data):
        self.data = data.data
        self.nr_samples = data.nr_samples
        self.gene_dict = data.gene_dict
        self.annot_index = data.annot_index

    def mean(self, idx, cat_=None, func_=None):
        # returns the mean value of the feature of index idx, across either all
        # samples, or samples with annotation cat_
        # the short id of the feature can be given instead of the index
        if type(idx) == str:
            idx = self.gene_dict[idx]
        if not func_:
            func_ = np.mean
        if not cat_:
            return func_(self.data[:, idx])
        else:
            return func_([self.data[i_, idx] for i_ in self.annot_index[cat_]])

    def std(self, idx, cat_=None):
        # returns the standard deviation of the feature of index idx, across either all
        # samples, or samples with annotation cat_
        # the short id of the feature can be given instead of the index
        return self.meangene(idx, cat_, np.std)

    def plot(self, idx, cat_=None, v_min=None, v_max=None):
        # plots the value of the feature of index idx for all samples
        # if cat_ is not None the samples of annotation cat_ have a different color
        # the short id of the feature can be given instead of the index
        if type(idx) == str:
            idx = self.gene_dict[idx]
        y = self.data[:, idx]
        if v_min is not None and v_max is not None:
            y = np.clip(y, v_min, v_max)
        x = np.arange(0, self.nr_samples) / self.nr_samples
        plt.scatter(x, y, s=1)
        if cat_:

            y = [self.data[i_, idx] for i_ in self.annot_index[cat_]]
            if v_min is not None and v_max is not None:
                y = np.clip(y, v_min, v_max)
            x = np.array(self.annot_index[cat_]) / self.nr_samples
            plt.scatter(x, y, s=1)
        plt.show()


def confusion_matrix(classifier, data_test, target_test):
    nr_neg = len(np.where(target_test == 0)[0])
    nr_pos = len(np.where(target_test == 1)[0])
    result = classifier.predict(data_test) - target_test
    fp = len(np.where(result == 1)[0])
    fn = len(np.where(result == -1)[0])
    tp = nr_pos - fn
    tn = nr_neg - fp
    return np.array([[tp, fp], [fn, tn]])


def matthews_coef(confusion_m):
    tp = confusion_m[0, 0]
    fp = confusion_m[0, 1]
    fn = confusion_m[1, 0]
    tn = confusion_m[1, 1]
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator == 0:
        denominator = 1
    mcc = (tp * tn - fp * fn) / np.sqrt(denominator)
    return mcc


def feature_selection_from_list(
    data,
    annotation,
    feature_indices,
):
    f_indices = np.zeros_like(feature_indices, dtype=np.int64)
    for i in range(len(f_indices)):
        if type(feature_indices[i]) == str or type(feature_indices[i]) == np.str_:
            f_indices[i] = data.gene_dict[feature_indices[i]]
        else:
            f_indices[i] = feature_indices[i]
    train_indices = sum(data.annot_index_train.values(), [])
    test_indices = sum(data.annot_index_test.values(), [])
    data_train = np.take(
        np.take(data.data.transpose(), f_indices, axis=0),
        train_indices,
        axis=1,
    ).transpose()
    target_train = np.zeros(data.nr_samples)
    target_train[data.annot_index_train[annotation]] = 1.0
    target_train = np.take(target_train, train_indices, axis=0)
    data_test = np.take(
        np.take(data.data.transpose(), f_indices, axis=0),
        test_indices,
        axis=1,
    ).transpose()
    target_test = np.zeros(data.nr_samples)
    target_test[data.annot_index_test[annotation]] = 1.0
    target_test = np.take(target_test, test_indices, axis=0)
    return (
        feature_indices,
        train_indices,
        test_indices,
        data_train,
        target_train,
        data_test,
        target_test,
    )


def naive_feature_selection(
    data,
    annotation,
    selection_size,
):
    feature_indices = np.array(
        data.expressions_on_training_sets_argsort[annotation][
            : (selection_size // 2)
        ].tolist()
        + data.expressions_on_training_sets_argsort[annotation][
            -(selection_size - selection_size // 2) :
        ].tolist()
    )
    return feature_selection_from_list(data, annotation, feature_indices)
