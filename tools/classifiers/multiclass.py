import numpy as np


class ScoreBasedMulticlass:
    def __init__(self):
        self.all_annotations = None
        self.binary_classifiers = {}

    def predict(self, x):
        scores = {}
        for annot in self.all_annotations:
            scores[annot] = self.binary_classifiers[annot].score(x)
        predictions = np.argmax(
            [scores[annot] for annot in self.all_annotations], axis=0
        )
        return np.array([self.all_annotations[i] for i in predictions])

    def save(self, fpath):
        for annot in self.all_annotations:
            sdir = fpath + "/" + str(annot).replace(" ", "_")
            if annot in self.binary_classifiers:
                self.binary_classifiers[annot].save(sdir)

    def load(self, fpath):
        for annot in self.all_annotations:
            sdir = fpath + "/" + str(annot).replace(" ", "_")
            if annot in self.binary_classifiers:
                if not self.binary_classifiers[annot].load(sdir):
                    return False
        return True
