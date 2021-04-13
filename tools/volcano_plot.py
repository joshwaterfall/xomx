import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression
from IPython import embed as e


class VolcanoPlot:
    def __init__(self, data, annotation, threshold=1e-5):
        self.data = data
        self.annotation = annotation
        self.threshold = threshold
        self.log2_foldchange = None
        self.log10_pvalues = None
        self.ok_data = None
        self.ok_target = None
        self.ok_indices = None

    def init(self):
        reference_values = self.data.mean_expressions
        on_annotation_values = (
            self.data.expressions_on_training_sets[self.annotation]
            * self.data.std_expressions
            + self.data.mean_expressions
        )
        self.ok_indices = np.where(reference_values > self.threshold)[0]
        reference_values = reference_values[self.ok_indices]
        on_annotation_values = on_annotation_values[self.ok_indices]
        ok2_indices = np.where(on_annotation_values > self.threshold)[0]
        reference_values = reference_values[ok2_indices]
        on_annotation_values = on_annotation_values[ok2_indices]
        self.ok_indices = self.ok_indices[ok2_indices]
        self.log2_foldchange = np.log2(on_annotation_values / reference_values)
        self.ok_data = np.take(
            self.data.data.transpose(), self.ok_indices, axis=0
        ).transpose()
        self.ok_target = np.zeros(self.data.nr_samples)
        self.ok_target[self.data.annot_index_train[self.annotation]] = 1.0
        self.ok_target[self.data.annot_index_test[self.annotation]] = 1.0
        fscores, pvalues = f_regression(self.ok_data, self.ok_target)
        self.log10_pvalues = -np.log10(pvalues + 1e-500)

    def plot(self, feature_list=[], save_dir=None):
        fig, ax = plt.subplots()
        e()
        # colors = np.zeros(len(self.ok_indices))
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        sc = ax.scatter(
            self.log2_foldchange, self.log10_pvalues, c="gray", cm="nipy_spectral", s=5
        )
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
            text = "{}".format(self.data.transcripts[ind["ind"][0]])
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
            plt.savefig(save_dir + "/volcano_plot.png", dpi=200)
        else:
            plt.show()
