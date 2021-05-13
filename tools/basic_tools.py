import os
import numpy as np
import re
import matplotlib.pyplot as plt
import umap

# from IPython import embed as e


class RNASeqData:
    """
    Attributes (None if they do not exist):

    save_id -> directory where data is saved

    sample_ids -> array of IDs: the i-th sample has ID sample_ids[i] (starting at 0)

    sample_indices -> sample of ID "#" has index sample_indices["#"]

    sample_annotations -> the i-th sample has annotation sample_annotations[i]

    sample_origins -> sample_origins[i] is a string characterizing the dataset of
                      origin for the i-th sample

    sample_origins_per_annotation -> sample_origins_per_annotation["#"] is the set
                                     of different origins for all the samples of
                                     annotation "#"

    all_annotations -> list of all annotations

    sample_indices_per_annotation -> sample_indices_per_annotation["#"] is the list
                                     of indices of the samples of annotation "#"

    nr_features -> the total number of features for each sample

    nr_samples -> the total number of samples

    feature_names -> the i-th feature name is feature_names[i]; it should be a string
                     of the form "<long_featureID>|<feature_shortname>"

    mean_expressions -> mean_expression[i] is the mean value of the i-th feature
                            accross all samples

    std_expressions -> similar to mean_expression but standard deviation instead of mean

    train_indices_per_annotation -> annot_index["#"] is the list of indices of the
                                    samples of annotation "#" which belong to the
                                    training set

    test_indices_per_annotation -> annot_index["#"] is the list of indices of the
                                   samples of annotation "#" which belong to the
                                   validation set

    feature_shortnames_ref -> if features have short IDs: feature_shortnames_ref["#"]
                              is the index of the feature of short name "#"

    std_values_on_training_sets -> std_values_on_training_sets["#"][j] is the mean
                                   value of the j-th feature, normalized by mean and
                                   std_dev, across all samples of annotation "#"
                                   belonging to the training set ; it is useful to
                                   determine whether a transcript is up-regulated
                                   for a given diagnosis (positive value), or
                                   down-regulated (negative value)

    std_values_on_training_sets_argsort -> std_values_on_training_sets_argsort["#"]
                                           is the list of feature indices sorted by
                                           decreasing value in
                                           std_values_on_training_sets["#"]

    epsilon_shift -> the value of the shift used for log-normalization of the data

    maxlog -> maximum value of the log data; it is a parameter computed during
              log-normalization

    raw_data -> raw_data[i, j]: value of the j-th feature of the i-th sample

    std_data -> data normalized by mean and standard deviation; the original value is:
                raw_data[i, j] == std_data[i, j] * std_expression[j]
                                  + mean_expression[j]

    log_data -> log-normalized values; the original value is:
                raw_data[i, j] == np.exp(
                                    log_data[i,j] * (maxlog - np.log(epsilon_shift)
                                  ) + np.log(epsilon_shift)) - epsilon_shift

    normalization_type -> if normalization_type=="raw", then data = raw_data
                          if normalization_type=="std", then data = std_data
                          if normalization_type=="log", then data = log_data

    nr_non_zero_features -> nr_non_zero_features[i] is, for the i-th sample, the number
                            of features with positive values (in raw data)

    nr_non_zero_samples -> nr_non_zero_samples[i] is the number of samples with positive
                           values on the i-th feature (in raw data)

    total_sums -> total_sums[i] is, for the i-th sample, the sum of values (in raw data)
                  accross all features
    """

    def __init__(self):
        self.save_dir = None
        self.sample_ids = None
        self.sample_indices = None
        self.sample_annotations = None
        self.sample_origins = None
        self.sample_origins_per_annotation = None
        self.all_annotations = None
        self.sample_indices_per_annotation = None
        self.nr_features = None
        self.nr_samples = None
        self.feature_names = None
        self.mean_expressions = None
        self.std_expressions = None
        self.data = None
        self.raw_data = None
        self.log_data = None
        self.std_data = None
        self.train_indices_per_annotation = None
        self.test_indices_per_annotation = None
        self.feature_shortnames_ref = None
        self.std_values_on_training_sets = None
        self.std_values_on_training_sets_argsort = None
        self.epsilon_shift = None
        self.maxlog = None
        self.normalization_type = None
        self.nr_non_zero_features = None
        self.nr_non_zero_samples = None
        self.total_sums = None

    def save(self):
        if not (os.path.exists(self.save_dir)):
            os.makedirs(self.save_dir, exist_ok=True)
        if self.nr_features is not None:
            np.save(self.save_dir + "nr_features.npy", self.nr_features)
            print("Saved: " + self.save_dir + "nr_features.npy")
        if self.nr_samples is not None:
            np.save(self.save_dir + "nr_samples.npy", self.nr_samples)
            print("Saved: " + self.save_dir + "nr_samples.npy")
        if self.sample_ids is not None:
            np.save(self.save_dir + "sample_ids.npy", self.sample_ids)
            print("Saved: " + self.save_dir + "sample_ids.npy")
        if self.sample_indices is not None:
            np.save(self.save_dir + "sample_indices.npy", self.sample_indices)
            print("Saved: " + self.save_dir + "sample_indices.npy")
        if self.sample_indices_per_annotation is not None:
            np.save(
                self.save_dir + "sample_indices_per_annotation.npy",
                self.sample_indices_per_annotation,
            )
            print("Saved: " + self.save_dir + "sample_indices_per_annotation.npy")
        if self.sample_annotations is not None:
            np.save(self.save_dir + "sample_annotations.npy", self.sample_annotations)
            print("Saved: " + self.save_dir + "sample_annotations.npy")
        if self.all_annotations is not None:
            np.save(self.save_dir + "all_annotations.npy", self.all_annotations)
            print("Saved: " + self.save_dir + "all_annotations.npy")
        if self.feature_names is not None:
            np.save(self.save_dir + "feature_names.npy", self.feature_names)
            print("Saved: " + self.save_dir + "feature_names.npy")
        if self.mean_expressions is not None:
            np.save(self.save_dir + "mean_expressions.npy", self.mean_expressions)
            print("Saved: " + self.save_dir + "mean_expressions.npy")
        if self.std_expressions is not None:
            np.save(self.save_dir + "std_expressions.npy", self.std_expressions)
            print("Saved: " + self.save_dir + "std_expressions.npy")
        if self.sample_origins is not None:
            np.save(self.save_dir + "sample_origins.npy", self.sample_origins)
            print("Saved: " + self.save_dir + "sample_origins.npy")
        if self.sample_origins_per_annotation is not None:
            np.save(
                self.save_dir + "sample_origins_per_annotation.npy",
                self.sample_origins_per_annotation,
            )
            print("Saved: " + self.save_dir + "sample_origins_per_annotations.npy")
        if self.train_indices_per_annotation is not None:
            np.save(
                self.save_dir + "train_indices_per_annotation.npy",
                self.train_indices_per_annotation,
            )
            print("Saved: " + self.save_dir + "train_indices_per_annotation.npy")
        if self.test_indices_per_annotation is not None:
            np.save(
                self.save_dir + "test_indices_per_annotation.npy",
                self.test_indices_per_annotation,
            )
            print("Saved: " + self.save_dir + "test_indices_per_annotation.npy")
        if self.std_values_on_training_sets is not None:
            np.save(
                self.save_dir + "std_values_on_training_sets.npy",
                self.std_values_on_training_sets,
            )
            print("Saved: " + self.save_dir + "std_values_on_training_sets.npy")
        if self.std_values_on_training_sets_argsort is not None:
            np.save(
                self.save_dir + "std_values_on_training_sets_argsort.npy",
                self.std_values_on_training_sets_argsort,
            )
            print("Saved: " + self.save_dir + "std_values_on_training_sets_argsort.npy")
        if self.epsilon_shift is not None:
            np.save(self.save_dir + "epsilon_shift.npy", self.epsilon_shift)
            print("Saved: " + self.save_dir + "epsilon_shift.npy")
        if self.maxlog is not None:
            np.save(self.save_dir + "maxlog.npy", self.maxlog)
            print("Saved: " + self.save_dir + "maxlog.npy")
        if self.feature_shortnames_ref is not None:
            np.save(
                self.save_dir + "feature_shortnames_ref.npy",
                self.feature_shortnames_ref,
            )
            print("Saved: " + self.save_dir + "feature_shortnames_ref.npy")
        if self.nr_non_zero_features is not None:
            np.save(
                self.save_dir + "nr_non_zero_features.npy",
                self.nr_non_zero_features,
            )
            print("Saved: " + self.save_dir + "nr_non_zero_features.npy")
        if self.nr_non_zero_samples is not None:
            np.save(
                self.save_dir + "nr_non_zero_samples.npy",
                self.nr_non_zero_samples,
            )
            print("Saved: " + self.save_dir + "nr_non_zero_samples.npy")
        if self.total_sums is not None:
            np.save(
                self.save_dir + "total_sums.npy",
                self.total_sums,
            )
            print("Saved: " + self.save_dir + "total_sums.npy")
        if self.raw_data is not None:
            fp_data = np.memmap(
                self.save_dir + "raw_data.bin",
                dtype="float32",
                mode="w+",
                shape=(self.nr_features, self.nr_samples),
            )
            fp_data[:] = self.raw_data.transpose()[:]
            del fp_data
            print("Saved: " + self.save_dir + "raw_data.bin")
        if self.std_data is not None:
            fp_data = np.memmap(
                self.save_dir + "std_data.bin",
                dtype="float32",
                mode="w+",
                shape=(self.nr_features, self.nr_samples),
            )
            fp_data[:] = self.std_data.transpose()[:]
            del fp_data
            print("Saved: " + self.save_dir + "std_data.bin")
        if self.log_data is not None:
            fp_data = np.memmap(
                self.save_dir + "log_data.bin",
                dtype="float32",
                mode="w+",
                shape=(self.nr_features, self.nr_samples),
            )
            fp_data[:] = self.log_data.transpose()[:]
            del fp_data
            print("Saved: " + self.save_dir + "log_data.bin")

    def load(self, normalization_types_list):
        assert self.save_dir is not None
        if os.path.exists(self.save_dir + "nr_features.npy"):
            self.nr_features = np.load(
                self.save_dir + "nr_features.npy", allow_pickle=True
            ).item()
        if os.path.exists(self.save_dir + "nr_samples.npy"):
            self.nr_samples = np.load(
                self.save_dir + "nr_samples.npy", allow_pickle=True
            ).item()
        if os.path.exists(self.save_dir + "sample_ids.npy"):
            self.sample_ids = np.load(
                self.save_dir + "sample_ids.npy", allow_pickle=True
            )
        if os.path.exists(self.save_dir + "sample_indices.npy"):
            self.sample_indices = np.load(
                self.save_dir + "sample_indices.npy", allow_pickle=True
            ).item()
        if os.path.exists(self.save_dir + "sample_indices_per_annotation.npy"):
            self.sample_indices_per_annotation = np.load(
                self.save_dir + "sample_indices_per_annotation.npy", allow_pickle=True
            ).item()
        if os.path.exists(self.save_dir + "sample_annotations.npy"):
            self.sample_annotations = np.load(
                self.save_dir + "sample_annotations.npy", allow_pickle=True
            )
        if os.path.exists(self.save_dir + "all_annotations.npy"):
            self.all_annotations = np.load(
                self.save_dir + "all_annotations.npy", allow_pickle=True
            )
        if os.path.exists(self.save_dir + "feature_names.npy"):
            self.feature_names = np.load(
                self.save_dir + "feature_names.npy", allow_pickle=True
            )
        if os.path.exists(self.save_dir + "mean_expressions.npy"):
            self.mean_expressions = np.load(
                self.save_dir + "mean_expressions.npy", allow_pickle=True
            )
        if os.path.exists(self.save_dir + "std_expressions.npy"):
            self.std_expressions = np.load(
                self.save_dir + "std_expressions.npy", allow_pickle=True
            )
        if os.path.exists(self.save_dir + "feature_shortnames_ref.npy"):
            self.feature_shortnames_ref = np.load(
                self.save_dir + "feature_shortnames_ref.npy", allow_pickle=True
            ).item()
        if os.path.exists(self.save_dir + "sample_origins.npy"):
            self.sample_origins = np.load(
                self.save_dir + "sample_origins.npy", allow_pickle=True
            )
        if os.path.exists(self.save_dir + "sample_origins_per_annotation.npy"):
            self.sample_origins_per_annotation = np.load(
                self.save_dir + "sample_origins_per_annotation.npy", allow_pickle=True
            ).item()
        if os.path.exists(self.save_dir + "train_indices_per_annotation.npy"):
            self.train_indices_per_annotation = np.load(
                self.save_dir + "train_indices_per_annotation.npy", allow_pickle=True
            ).item()
        if os.path.exists(self.save_dir + "test_indices_per_annotation.npy"):
            self.test_indices_per_annotation = np.load(
                self.save_dir + "test_indices_per_annotation.npy", allow_pickle=True
            ).item()
        if os.path.exists(self.save_dir + "std_values_on_training_sets.npy"):
            self.std_values_on_training_sets = np.load(
                self.save_dir + "std_values_on_training_sets.npy", allow_pickle=True
            ).item()
        if os.path.exists(self.save_dir + "std_values_on_training_sets_argsort.npy"):
            self.std_values_on_training_sets_argsort = np.load(
                self.save_dir + "std_values_on_training_sets_argsort.npy",
                allow_pickle=True,
            ).item()
        if os.path.exists(self.save_dir + "epsilon_shift.npy"):
            self.epsilon_shift = np.load(
                self.save_dir + "epsilon_shift.npy", allow_pickle=True
            ).item()
        if os.path.exists(self.save_dir + "maxlog.npy"):
            self.maxlog = np.load(
                self.save_dir + "maxlog.npy", allow_pickle=True
            ).item()
        if os.path.exists(self.save_dir + "nr_non_zero_samples.npy"):
            self.nr_non_zero_samples = np.load(
                self.save_dir + "nr_non_zero_samples.npy", allow_pickle=True
            )
        if os.path.exists(self.save_dir + "nr_non_zero_features.npy"):
            self.nr_non_zero_features = np.load(
                self.save_dir + "nr_non_zero_features.npy", allow_pickle=True
            )
        if os.path.exists(self.save_dir + "total_sums.npy"):
            self.total_sums = np.load(
                self.save_dir + "total_sums.npy", allow_pickle=True
            )
        if (
            os.path.exists(self.save_dir + "raw_data.bin")
            and "raw" in normalization_types_list
        ):
            self.raw_data = np.array(
                np.memmap(
                    self.save_dir + "raw_data.bin",
                    dtype="float32",
                    mode="r",
                    shape=(self.nr_features, self.nr_samples),
                )
            ).transpose()
        if (
            os.path.exists(self.save_dir + "std_data.bin")
            and "std" in normalization_types_list
        ):
            self.std_data = np.array(
                np.memmap(
                    self.save_dir + "std_data.bin",
                    dtype="float32",
                    mode="r",
                    shape=(self.nr_features, self.nr_samples),
                )
            ).transpose()
        if (
            os.path.exists(self.save_dir + "log_data.bin")
            and "log" in normalization_types_list
        ):
            self.log_data = np.array(
                np.memmap(
                    self.save_dir + "log_data.bin",
                    dtype="float32",
                    mode="r",
                    shape=(self.nr_features, self.nr_samples),
                )
            ).transpose()
        if len(normalization_types_list) > 0:
            if normalization_types_list[0] == "raw":
                print('Normalization type: "raw"')
                self.data = self.raw_data
                self.normalization_type = "raw"
            elif normalization_types_list[0] == "std":
                self.data = self.std_data
                self.normalization_type = "std"
                print('Normalization type: "std"')
            elif normalization_types_list[0] == "log":
                self.data = self.log_data
                self.normalization_type = "log"
                print('Normalization type: "log"')

    def compute_sample_indices(self):
        assert self.sample_ids is not None
        self.sample_indices = {}
        for i, s_id in enumerate(self.sample_ids):
            self.sample_indices[s_id] = i

    def compute_sample_indices_per_annotation(self):
        assert self.sample_annotations is not None
        self.sample_indices_per_annotation = {}
        for i, annot in enumerate(self.sample_annotations):
            self.sample_indices_per_annotation.setdefault(annot, [])
            self.sample_indices_per_annotation[annot].append(i)

    def compute_all_annotations(self):
        assert self.sample_annotations is not None
        self.all_annotations = np.array(list(dict.fromkeys(self.sample_annotations)))

    def compute_sample_origins_per_annotation(self):
        assert self.sample_annotations is not None and self.sample_origins is not None
        self.sample_origins_per_annotation = {}
        for i, annot in enumerate(self.sample_annotations):
            self.sample_origins_per_annotation.setdefault(annot, set()).add(
                self.sample_origins[i]
            )

    def compute_mean_expressions(self):
        assert self.raw_data is not None and self.nr_features is not None
        self.mean_expressions = [
            np.mean(self.raw_data[:, i]) for i in range(self.nr_features)
        ]

    def compute_std_expressions(self):
        assert self.raw_data is not None and self.nr_features is not None
        self.std_expressions = [
            np.std(self.raw_data[:, i]) for i in range(self.nr_features)
        ]

    def compute_std_data(self):
        assert (
            self.raw_data is not None
            and self.mean_expressions is not None
            and self.std_expressions is not None
            and self.nr_features is not None
        )
        self.std_data = np.copy(self.raw_data)
        for i in range(self.nr_features):
            if self.std_expressions[i] == 0.0:
                self.std_data[:, i] = 0.0
            else:
                self.std_data[:, i] = (
                    self.std_data[:, i] - self.mean_expressions[i]
                ) / self.std_expressions[i]
            # for j in range(self.nr_samples):
            #     if self.std_expressions[i] == 0.0:
            #         self.std_data[j, i] = 0.0
            #     else:
            #         self.std_data[j, i] = (
            #                                       self.std_data[j, i] -
            #                                       self.mean_expressions[i]
            #                               ) / self.std_expressions[i]

    def compute_log_data(self):
        assert (
            self.raw_data is not None
            and self.mean_expressions is not None
            and self.std_expressions is not None
            and self.nr_features is not None
        )
        self.log_data = np.copy(self.raw_data)
        self.epsilon_shift = 1.0
        for i in range(self.nr_features):
            self.log_data[:, i] = np.log(self.log_data[:, i] + self.epsilon_shift)
        self.maxlog = np.max(self.log_data)
        for i in range(self.nr_features):
            self.log_data[:, i] = (self.log_data[:, i] - np.log(self.epsilon_shift)) / (
                self.maxlog - np.log(self.epsilon_shift)
            )

    def compute_train_and_test_indices_per_annotation(self):
        assert self.sample_indices_per_annotation is not None
        self.train_indices_per_annotation = {}
        self.test_indices_per_annotation = {}
        for annot in self.sample_indices_per_annotation:
            idxs = np.random.permutation(self.sample_indices_per_annotation[annot])
            cut = len(idxs) // 4 + 1
            self.test_indices_per_annotation[annot] = idxs[:cut]
            self.train_indices_per_annotation[annot] = idxs[cut:]

    def compute_feature_shortnames_ref(self):
        assert self.feature_names is not None
        self.feature_shortnames_ref = {}
        for i, elt in enumerate(self.feature_names):
            self.feature_shortnames_ref[elt.split("|")[1]] = i

    def compute_std_values_on_training_sets(self):
        assert (
            self.all_annotations is not None
            and self.nr_features is not None
            and self.std_data is not None
            and self.train_indices_per_annotation is not None
        )
        self.std_values_on_training_sets = {}
        for annot in self.all_annotations:
            self.std_values_on_training_sets[annot] = []
            for i in range(self.nr_features):
                self.std_values_on_training_sets[annot].append(
                    np.mean(
                        [
                            self.std_data[j, i]
                            for j in self.train_indices_per_annotation[annot]
                        ]
                    )
                )

    def compute_std_values_on_training_sets_argsort(self):
        assert (
            self.all_annotations is not None
            and self.std_values_on_training_sets is not None
        )
        self.std_values_on_training_sets_argsort = {}
        for annot in self.all_annotations:
            self.std_values_on_training_sets_argsort[annot] = np.argsort(
                self.std_values_on_training_sets[annot]
            )[::-1]

    def compute_nr_non_zero_features(self):
        assert self.raw_data is not None and self.nr_samples is not None
        self.nr_non_zero_features = np.empty((self.nr_samples,), dtype=int)
        for i in range(self.nr_samples):
            self.nr_non_zero_features[i] = len(np.where(self.raw_data[i, :] > 0.0)[0])

    def compute_nr_non_zero_samples(self):
        assert self.raw_data is not None and self.nr_features is not None
        self.nr_non_zero_samples = np.empty((self.nr_features,), dtype=int)
        for i in range(self.nr_features):
            self.nr_non_zero_samples[i] = len(np.where(self.raw_data[:, i] > 0.0)[0])

    def compute_total_sums(self):
        assert self.raw_data is not None and self.nr_samples is not None
        self.total_sums = np.empty((self.nr_samples,), dtype=float)
        for i in range(self.nr_samples):
            self.total_sums[i] = np.sum(self.raw_data[i, :])

    def reduce_samples(self, idx_list):
        if self.nr_samples is not None:
            self.nr_samples = len(idx_list)
        if self.sample_ids is not None:
            self.sample_ids = np.take(self.sample_ids, idx_list)
        if self.sample_annotations is not None:
            self.sample_annotations = np.take(self.sample_annotations, idx_list)
        if self.sample_indices is not None:
            self.compute_sample_indices()
        if self.all_annotations is not None:
            self.compute_all_annotations()
        if self.sample_indices_per_annotation is not None:
            self.compute_sample_indices_per_annotation()
        if self.sample_origins is not None:
            self.sample_origins = np.take(self.sample_origins, idx_list)
        if self.sample_origins_per_annotation is not None:
            self.compute_sample_indices_per_annotation()
        if self.raw_data is not None:
            self.raw_data = np.take(self.raw_data, idx_list, axis=0)
        if self.std_data is not None:
            self.std_data = np.take(self.std_data, idx_list, axis=0)
        if self.log_data is not None:
            self.log_data = np.take(self.log_data, idx_list, axis=0)
        if self.mean_expressions is not None:
            self.compute_mean_expressions()
        if self.std_expressions is not None:
            self.compute_std_expressions()
        if (
            self.train_indices_per_annotation is not None
            and self.test_indices_per_annotation is not None
        ):
            self.compute_train_and_test_indices_per_annotation()
        if self.std_values_on_training_sets is not None:
            self.compute_std_values_on_training_sets()
        if self.std_values_on_training_sets_argsort is not None:
            self.compute_std_values_on_training_sets_argsort()
        if self.nr_non_zero_features is not None:
            self.compute_nr_non_zero_features()
        if self.nr_non_zero_samples is not None:
            self.compute_nr_non_zero_samples()
        if self.total_sums is not None:
            self.compute_total_sums()

    def reduce_features(self, idx_list):
        if self.nr_features is not None:
            self.nr_features = len(idx_list)
        if self.feature_names is not None:
            self.feature_names = np.take(self.feature_names, idx_list)
        if self.mean_expressions is not None:
            self.mean_expressions = np.take(self.mean_expressions, idx_list)
        if self.std_expressions is not None:
            self.std_expressions = np.take(self.std_expressions, idx_list)
        if self.feature_shortnames_ref is not None:
            self.compute_feature_shortnames_ref()
        if self.raw_data is not None:
            self.raw_data = np.take(
                self.raw_data.transpose(), idx_list, axis=0
            ).transpose()
        if self.std_data is not None:
            self.std_data = np.take(
                self.std_data.transpose(), idx_list, axis=0
            ).transpose()
        if self.log_data is not None:
            self.log_data = np.take(
                self.log_data.transpose(), idx_list, axis=0
            ).transpose()
        if self.normalization_type is not None:
            if self.normalization_type == "raw":
                self.data = self.raw_data
            elif self.normalization_type == "std":
                self.data = self.std_data
            elif self.normalization_type == "log":
                self.data = self.log_data
        if (
            self.all_annotations is not None
            and self.std_values_on_training_sets is not None
        ):
            for cat in self.all_annotations:
                self.std_values_on_training_sets[cat] = list(
                    np.take(self.std_values_on_training_sets[cat], idx_list)
                )
            self.compute_std_values_on_training_sets_argsort()
        if self.nr_non_zero_features is not None:
            self.compute_nr_non_zero_features()
        if self.nr_non_zero_samples is not None:
            self.compute_nr_non_zero_samples()
        if self.total_sums is not None:
            self.compute_total_sums()

    def percentage_feature_set(self, idx_list, sample_idx=None):
        """computes the sum of values (in raw data), across all samples or for one
        given sample, for features of indices in idx_list, divided by the sum of values
        for all the features"""
        assert self.raw_data is not None
        if sample_idx:
            return np.sum(self.raw_data[sample_idx, idx_list]) / np.sum(
                self.raw_data[sample_idx, :]
            )
        else:
            return np.sum(self.raw_data[:, idx_list]) / np.sum(self.raw_data)

    def regex_search(self, rexpr):
        """tests for every feature name whether it matches the regular expression
        rexpr; returns the list of indices of the features that do match
        """
        return np.where([re.search(rexpr, s) for s in self.feature_names])[0]

    def feature_mean(self, idx, cat_=None, func_=None):
        """returns the mean value of the feature of index idx, across either all
        samples, or samples with annotation cat_
        the short name of the feature can be given instead of the index"""
        if type(idx) == str:
            idx = self.feature_shortnames_ref[idx]
        if not func_:
            func_ = np.mean
        if not cat_:
            return func_(self.data[:, idx])
        else:
            return func_(
                [self.data[i_, idx] for i_ in self.sample_indices_per_annotation[cat_]]
            )

    def feature_std(self, idx, cat_=None):
        """returns the standard deviation of the feature of index idx, across either all
        samples, or samples with annotation cat_;
        the short name of the feature can be given instead of the index"""
        return self.feature_mean(idx, cat_, np.std)

    def feature_plot(self, idx, cat_=None, v_min=None, v_max=None):
        """plots the value of the feature of index idx for all samples;
        if cat_ is not None the samples of annotation cat_ have a different color
        the short name of the feature can be given instead of the index"""
        if type(idx) == str:
            idx = self.feature_shortnames_ref[idx]
        y = self.data[:, idx]
        if v_min is not None and v_max is not None:
            y = np.clip(y, v_min, v_max)
        x = np.arange(0, self.nr_samples) / self.nr_samples
        plt.scatter(x, y, s=1)
        if cat_:

            y = [self.data[i_, idx] for i_ in self.sample_indices_per_annotation[cat_]]
            if v_min is not None and v_max is not None:
                y = np.clip(y, v_min, v_max)
            x = np.array(self.sample_indices_per_annotation[cat_]) / self.nr_samples
            plt.scatter(x, y, s=1)
        plt.show()

    # def function_plot(self, func_=np.identity, cat_=None):
    #     """plots the value of a function on all samples (the function must take sample
    #     indices in input);
    #     if cat_ is not None the samples of annotation cat_ have a different color"""
    #     y = [func_(i) for i in range(self.nr_samples)]
    #     x = np.arange(0, self.nr_samples) / self.nr_samples
    #     fig, ax = plt.subplots()
    #     parts = ax.violinplot(
    #         y,
    #         [0.5],
    #         points=60,
    #         widths=1.0,
    #         showmeans=False,
    #         showextrema=False,
    #         showmedians=False,
    #         bw_method=0.5,
    #     )
    #     for pc in parts["bodies"]:
    #         pc.set_facecolor("#D43F3A")
    #         pc.set_edgecolor("grey")
    #         pc.set_alpha(0.7)
    #     ax.scatter(x, y, s=1)
    #
    #     if cat_:
    #         y = [func_(i_) for i_ in self.sample_indices_per_annotation[cat_]]
    #         x = np.array(self.sample_indices_per_annotation[cat_]) / self.nr_samples
    #         plt.scatter(x, y, s=1)
    #     plt.show()

    def function_scatter(
        self,
        func1_=np.identity,
        func2_=np.identity,
        sof_="samples",
        cat_=None,
        violinplot_=False,
    ):
        """displays a scatter plot, with coordinates computed by applying two
        functions (func1_ and func2_) to every sample or every feature, depending
        on the value of sof_ which must be either "samples" or "features"
        (both functions must take indices in input);
        if sof=="samples" and cat_ is not None the samples of annotation cat_ have
        a different color"""
        assert sof_ == "samples" or sof_ == "features"
        if sof_ == "samples":
            y = [func2_(i) for i in range(self.nr_samples)]
            x = [func1_(i) for i in range(self.nr_samples)]
        else:
            y = [func2_(i) for i in range(self.nr_features)]
            x = [func1_(i) for i in range(self.nr_features)]
        xmax = np.max(x)
        xmin = np.min(x)
        fig, ax = plt.subplots()
        if violinplot_:
            parts = ax.violinplot(
                y,
                [(xmax + xmin) / 2.0],
                points=60,
                widths=xmax - xmin,
                showmeans=False,
                showextrema=False,
                showmedians=False,
                bw_method=0.5,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor("#D43F3A")
                pc.set_edgecolor("grey")
                pc.set_alpha(0.7)
        scax = ax.scatter(x, y, s=1)
        ann = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(-100, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        ann.set_visible(False)

        def update_annot(ind, sc):
            pos = sc.get_offsets()[ind["ind"][0]]
            ann.xy = pos
            if sof_ == "samples":
                text = "{}".format(self.sample_ids[ind["ind"][0]])
            else:
                text = "{}".format(self.feature_names[ind["ind"][0]])
            ann.set_text(text)

        def hover(event):
            vis = ann.get_visible()
            if event.inaxes == ax:
                cont, ind = scax.contains(event)
                if cont:
                    update_annot(ind, scax)
                    ann.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        ann.set_visible(False)
                        fig.canvas.draw_idle()
            if event.inaxes == ax:
                cont, ind = scax.contains(event)
                if cont:
                    update_annot(ind, scax)
                    ann.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        ann.set_visible(False)
                        fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)

        if sof_ == "samples" and cat_:
            y = [func2_(i_) for i_ in self.sample_indices_per_annotation[cat_]]
            x = [func1_(i_) for i_ in self.sample_indices_per_annotation[cat_]]
            ax.scatter(x, y, s=1)
        plt.show()

    def function_plot(
        self, func_=lambda x: x, sof_="samples", cat_=None, violinplot_=True
    ):
        """plots the value of a function on every sample or every feature, depending
        on the value of sof_ which must be either "samples" or "features"
        (the function must take indices in input);
        if sof== "samples" and cat_ is not None the samples of annotation cat_ have a
        different color"""
        self.function_scatter(lambda x: x, func_, sof_, cat_, violinplot_)


# class FeatureTools:
#     def __init__(self, data):
#         self.data = data.data
#         self.nr_samples = data.nr_samples
#         self.gene_dict = data.feature_shortnames_ref
#         self.annot_index = data.sample_indices_per_annotation
#
#     def mean(self, idx, cat_=None, func_=None):
#         # returns the mean value of the feature of index idx, across either all
#         # samples, or samples with annotation cat_
#         # the short id of the feature can be given instead of the index
#         if type(idx) == str:
#             idx = self.gene_dict[idx]
#         if not func_:
#             func_ = np.mean
#         if not cat_:
#             return func_(self.data[:, idx])
#         else:
#             return func_([self.data[i_, idx] for i_ in self.annot_index[cat_]])
#
#     def std(self, idx, cat_=None):
#         # returns the standard deviation of the feature of index idx, across either
#         # all samples, or samples with annotation cat_;
#         # the short id of the feature can be given instead of the index
#         return self.mean(idx, cat_, np.std)
#
#     def plot(self, idx, cat_=None, v_min=None, v_max=None):
#         # plots the value of the feature of index idx for all samples
#         # if cat_ is not None the samples of annotation cat_ have a different color
#         # the short id of the feature can be given instead of the index
#         if type(idx) == str:
#             idx = self.gene_dict[idx]
#         y = self.data[:, idx]
#         if v_min is not None and v_max is not None:
#             y = np.clip(y, v_min, v_max)
#         x = np.arange(0, self.nr_samples) / self.nr_samples
#         plt.scatter(x, y, s=1)
#         if cat_:
#             y = [self.data[i_, idx] for i_ in self.annot_index[cat_]]
#             if v_min is not None and v_max is not None:
#                 y = np.clip(y, v_min, v_max)
#             x = np.array(self.annot_index[cat_]) / self.nr_samples
#             plt.scatter(x, y, s=1)
#         plt.show()


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
            f_indices[i] = data.feature_shortnames_ref[feature_indices[i]]
        else:
            f_indices[i] = feature_indices[i]
    # train_indices = sum(data.train_indices_per_annotation.values(), [])
    # test_indices = sum(data.test_indices_per_annotation.values(), [])
    train_indices = np.concatenate(list(data.train_indices_per_annotation.values()))
    test_indices = np.concatenate(list(data.test_indices_per_annotation.values()))
    data_train = np.take(
        np.take(data.data.transpose(), f_indices, axis=0),
        train_indices,
        axis=1,
    ).transpose()
    target_train = np.zeros(data.nr_samples)
    target_train[data.train_indices_per_annotation[annotation]] = 1.0
    target_train = np.take(target_train, train_indices, axis=0)
    data_test = np.take(
        np.take(data.data.transpose(), f_indices, axis=0),
        test_indices,
        axis=1,
    ).transpose()
    target_test = np.zeros(data.nr_samples)
    target_test[data.test_indices_per_annotation[annotation]] = 1.0
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
        data.std_values_on_training_sets_argsort[annotation][
            : (selection_size // 2)
        ].tolist()
        + data.std_values_on_training_sets_argsort[annotation][
            -(selection_size - selection_size // 2) :
        ].tolist()
    )
    return feature_selection_from_list(data, annotation, feature_indices)


def plot_scores(data, scores, score_threshold, indices, annotation=None, save_dir=None):
    annot_colors = {}
    denom = len(data.all_annotations)
    for i, val in enumerate(data.all_annotations):
        if annotation:
            if val == annotation:
                annot_colors[val] = 0.0 / denom
            else:
                annot_colors[val] = (denom + i) / denom
        else:
            annot_colors[val] = i / denom

    samples_color = np.zeros(len(indices))
    for i in range(len(indices)):
        samples_color[i] = annot_colors[data.sample_annotations[indices[i]]]

    fig, ax = plt.subplots()
    if annotation:
        cm = "winter"
    else:
        cm = "nipy_spectral"
    sc = ax.scatter(np.arange(len(indices)), scores, c=samples_color, cmap=cm, s=5)
    ax.axhline(y=score_threshold, xmin=0, xmax=1, lw=1, ls="--", c="red")
    ann = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(-100, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    ann.set_visible(False)

    def update_annot(ind, sc):
        pos = sc.get_offsets()[ind["ind"][0]]
        ann.xy = pos
        text = "{}".format(data.sample_annotations[indices[ind["ind"][0]]])
        ann.set_text(text)

    def hover(event):
        vis = ann.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind, sc)
                ann.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    ann.set_visible(False)
                    fig.canvas.draw_idle()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind, sc)
                ann.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    ann.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir + "/plot.png", dpi=200)
    else:
        plt.show()


def umap_plot(
    data,
    x,
    indices,
    save_dir=None,
    metric="euclidean",
    min_dist=0.0,
    n_neighbors=120,
    random_state=42,
):
    reducer = umap.UMAP(
        metric=metric,
        min_dist=min_dist,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    print("Starting UMAP reduction...")
    reducer.fit(x)
    embedding = reducer.transform(x)
    print("Done.")

    all_colors = []

    def color_function(id_):
        label_ = data.sample_annotations[indices[id_]]
        type_ = data.sample_origins[indices[id_]]
        clo = (
            np.where(data.all_annotations == label_)[0],
            list(data.sample_origins_per_annotation[label_]).index(type_),
        )
        if clo in all_colors:
            return all_colors.index(clo)
        else:
            all_colors.append(clo)
            return len(all_colors) - 1

    def hover_function(id_):
        return "{} / {}".format(
            data.sample_annotations[indices[id_]],
            data.sample_origins[indices[id_]],
        )

    samples_color = np.empty_like(indices)
    for i in range(len(indices)):
        samples_color[i] = color_function(i)

    fig, ax = plt.subplots()

    sc = plt.scatter(
        embedding[:, 0], embedding[:, 1], c=samples_color, cmap="winter", s=5
    )
    plt.gca().set_aspect("equal", "datalim")

    ann = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(20, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    ann.set_visible(False)

    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        ann.xy = pos
        text = hover_function(ind["ind"][0])
        ann.set_text(text)

    def hover(event):
        vis = ann.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                ann.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    ann.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_dir + "/plot.png", dpi=200)
    else:
        plt.show()
