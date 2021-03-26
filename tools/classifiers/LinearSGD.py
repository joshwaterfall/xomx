import os
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np


class LinearSGD:
    def __init__(self, data, max_iter=1000, tol=1e-3):
        self.data = data
        self.clf = make_pipeline(
            StandardScaler(), SGDClassifier(max_iter=max_iter, tol=tol)
        )

    def fit(self, x, y):
        self.clf.fit(x, y)

    def fit_list(self, transcripts_list, annotation):
        transcripts_indices = np.copy(transcripts_list)
        for i in range(len(transcripts_indices)):
            if (
                type(transcripts_indices[i]) == str
                or type(transcripts_indices[i]) == np.str_
            ):
                transcripts_indices[i] = self.data.gene_dict[transcripts_indices[i]]
        transcripts_indices = np.array(transcripts_indices).astype(int)
        train_indices = sum(self.data.annot_index_train.values(), [])
        data_train = np.take(
            np.take(self.data.data.transpose(), transcripts_indices, axis=0),
            train_indices,
            axis=1,
        ).transpose()
        target_train = np.zeros(self.data.nr_samples)
        target_train[self.data.annot_index_train[annotation]] = 1.0
        target_train = np.take(target_train, train_indices, axis=0)
        self.clf.fit(data_train, target_train)

    def plot(self, x, indices, annotation=None, save_dir=None):
        res = self.clf.decision_function(x)
        annot_colors = {}
        denom = len(self.data.annot_values)
        for i, val in enumerate(self.data.annot_values):
            if annotation:
                if val == annotation:
                    annot_colors[val] = 0.0 / denom
                else:
                    annot_colors[val] = (denom + i) / denom
            else:
                annot_colors[val] = i / denom

        samples_color = np.zeros(len(indices))
        for i in range(len(indices)):
            samples_color[i] = annot_colors[
                self.data.annot_dict[self.data.samples_id[indices[i]]]
            ]

        fig, ax = plt.subplots()
        if annotation:
            cm = "winter"
        else:
            cm = "nipy_spectral"
        sc = ax.scatter(np.arange(len(indices)), res, c=samples_color, cmap=cm, s=5)

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
            text = "{}".format(
                self.data.annot_dict[self.data.samples_id[indices[ind["ind"][0]]]]
            )
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

    def plot_list(self, transcripts_list, annotation=None, save_dir=None):
        transcripts_indices = np.copy(transcripts_list)
        for i in range(len(transcripts_indices)):
            if (
                type(transcripts_indices[i]) == str
                or type(transcripts_indices[i]) == np.str_
            ):
                transcripts_indices[i] = self.data.gene_dict[transcripts_indices[i]]
        transcripts_indices = np.array(transcripts_indices).astype(int)
        test_indices = sum(self.data.annot_index_test.values(), [])
        data_test = np.take(
            np.take(self.data.data.transpose(), transcripts_indices, axis=0),
            test_indices,
            axis=1,
        ).transpose()
        self.plot(data_test, test_indices, annotation, save_dir)
