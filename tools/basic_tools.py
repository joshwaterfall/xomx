import os
import numpy as np
import re
import matplotlib.pyplot as plt
import umap


class RNASeqData:
    """
    Attributes (None if they do not exist):

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
                                    value of the j-th feature, normalized by mean and
                                    std_dev, across all samples of annotation "#"
                                    belonging to the training set ; it is useful to
                                    determine whether a transcript is up-regulated
                                    for a given diagnosis (positive value), or
                                    down-regulated (negative value)

    expressions_on_training_sets_argsort -> expressions_on_training_sets_argsort["#"]
                                            is the list of feature indices sorted by
                                            decreasing value in
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
            if the normalization is of type ( . - mean) / std_dev ("mean_std"):
              data[i, j] * std_expression[j] + mean_expression[j] is the original value
            if the normalization is of type log-norm ("log"):
              the original value is:
              np.exp( data[i,j] * (maxlog - np.log(epsilon_shift))
                     + np.log(epsilon_shift)) - epsilon_shift
            if no normalization ("raw"):
              data[i, j] is the original value

    non_zero_features -> for each sample, the list of features with positive counts (in
                         raw data)

    total_counts -> for each sample, the sum of counts (in raw data) accross all
                    features
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
        self.nr_non_zero_features = None
        self.total_counts = None

    def reduce_samples(self, idx_list):
        # TODO
        pass

    def compute_nr_non_zero_features(self):
        assert self.normalization_type == "raw"
        self.nr_non_zero_features = np.empty((self.nr_samples,), dtype=int)
        for i in range(self.nr_samples):
            self.nr_non_zero_features[i] = len(np.where(self.data[i, :] > 0.0)[0])

    def compute_total_counts(self):
        assert self.normalization_type == "raw"
        self.total_counts = np.empty((self.nr_samples,), dtype=float)
        for i in range(self.nr_samples):
            self.total_counts[i] = np.sum(self.data[i, :])

    def reduce_features(self, idx_list):
        self.nr_transcripts = len(idx_list)
        self.transcripts = np.take(self.transcripts, idx_list)
        self.mean_expressions = np.take(self.mean_expressions, idx_list)
        self.std_expressions = np.take(self.std_expressions, idx_list)
        self.data = np.take(self.data.transpose(), idx_list, axis=0).transpose()
        self.gene_dict = {}
        for i, elt in enumerate(self.transcripts):
            self.gene_dict[elt.split("|")[1]] = i
        if self.annot_values:
            for cat in self.annot_values:
                self.expressions_on_training_sets[cat] = list(
                    np.take(self.expressions_on_training_sets[cat], idx_list)
                )
            for cat in self.annot_values:
                self.expressions_on_training_sets_argsort[cat] = np.argsort(
                    self.expressions_on_training_sets[cat]
                )[::-1]
        self.total_counts = None
        self.nr_non_zero_features = None
        if self.normalization_type == "raw":
            self.compute_total_counts()
            self.compute_nr_non_zero_features()

    def percentage_feature_set(self, idx_list, sample_idx=None):
        """computes the sum of values, across all samples or for one given sample,
        for features of indices in idx_list, divided by the sum of values for all
        the features"""
        if sample_idx:
            return np.sum(self.data[sample_idx, idx_list]) / np.sum(
                self.data[sample_idx, :]
            )
        else:
            return np.sum(self.data[:, idx_list]) / np.sum(self.data)

    def regex_search(self, rexpr):
        """tests for every feature name whether it matches the regular expression
        rexpr; returns the list of indices of the features that do match
        """
        return np.where([re.search(rexpr, s) for s in self.transcripts])[0]

    def feature_mean(self, idx, cat_=None, func_=None):
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

    def feature_std(self, idx, cat_=None):
        # returns the standard deviation of the feature of index idx, across either all
        # samples, or samples with annotation cat_
        # the short id of the feature can be given instead of the index
        return self.feature_mean(idx, cat_, np.std)

    def feature_plot(self, idx, cat_=None, v_min=None, v_max=None):
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

    def function_plot(self, func_=np.identity, cat_=None):
        # plots the value of a function on all samples (the function must take sample
        # indices in input)
        # if cat_ is not None the samples of annotation cat_ have a different color
        y = [func_(i) for i in range(self.nr_samples)]
        x = np.arange(0, self.nr_samples) / self.nr_samples
        fig, ax = plt.subplots()
        parts = ax.violinplot(
            y,
            [0.5],
            points=60,
            widths=1.0,
            showmeans=False,
            showextrema=False,
            showmedians=False,
            bw_method=0.5,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("#D43F3A")
            pc.set_edgecolor("grey")
            pc.set_alpha(0.7)
        ax.scatter(x, y, s=1)

        if cat_:
            y = [func_(i_) for i_ in self.annot_index[cat_]]
            x = np.array(self.annot_index[cat_]) / self.nr_samples
            plt.scatter(x, y, s=1)
        plt.show()


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
        return self.mean(idx, cat_, np.std)

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


def plot_scores(data, scores, score_threshold, indices, annotation=None, save_dir=None):
    annot_colors = {}
    denom = len(data.annot_values)
    for i, val in enumerate(data.annot_values):
        if annotation:
            if val == annotation:
                annot_colors[val] = 0.0 / denom
            else:
                annot_colors[val] = (denom + i) / denom
        else:
            annot_colors[val] = i / denom

    samples_color = np.zeros(len(indices))
    for i in range(len(indices)):
        samples_color[i] = annot_colors[data.annot_dict[data.samples_id[indices[i]]]]

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
        text = "{}".format(data.annot_dict[data.samples_id[indices[ind["ind"][0]]]])
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
        label_ = data.annot_dict[data.samples_id[indices[id_]]]
        type_ = data.annot_types[data.samples_id[indices[id_]]]
        clo = (
            np.where(data.annot_values == label_)[0],
            list(data.annot_types_dict[label_]).index(type_),
        )
        if clo in all_colors:
            return all_colors.index(clo)
        else:
            all_colors.append(clo)
            return len(all_colors) - 1

    def hover_function(id_):
        return "{} / {}".format(
            data.annot_dict[data.samples_id[indices[id_]]],
            data.annot_types[data.samples_id[indices[id_]]],
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
