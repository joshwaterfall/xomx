import os
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from tools.basic_tools import confusion_matrix, naive_feature_selection, plot_scores
from joblib import dump, load

# from IPython import embed as e


class RFEExtraTrees:
    def __init__(
        self,
        data,
        annotation,
        init_selection_size=4000,
        n_estimators=450,
        random_state=0,
    ):
        self.data = data
        self.annotation = annotation
        self.init_selection_size = init_selection_size
        self.n_estimators = 450
        self.random_state = 0
        (
            self.current_feature_indices,
            self.train_indices,
            self.test_indices,
            self.data_train,
            self.target_train,
            self.data_test,
            self.target_test,
        ) = naive_feature_selection(
            self.data, self.annotation, self.init_selection_size
        )
        self.forest = None
        self.confusion_matrix = None
        self.log = []

    def init(self):
        self.forest = ExtraTreesClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state
        )
        self.forest.fit(self.data_train, self.target_train)
        self.confusion_matrix = confusion_matrix(
            self.forest, self.data_test, self.target_test
        )
        self.log.append(
            {
                "feature_indices": self.current_feature_indices,
                "confusion_matrix": self.confusion_matrix,
            }
        )

    def select_features(self, n):
        assert n <= self.data_train.shape[1]
        sorted_feats = np.argsort(self.forest.feature_importances_)[::-1]
        reduced_feats = list(sorted_feats[:n])
        self.current_feature_indices = np.take(
            self.current_feature_indices, reduced_feats, axis=0
        )
        self.data_train = np.take(
            self.data_train.transpose(), reduced_feats, axis=0
        ).transpose()
        self.data_test = np.take(
            self.data_test.transpose(), reduced_feats, axis=0
        ).transpose()
        self.forest = ExtraTreesClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state
        )
        self.forest.fit(self.data_train, self.target_train)
        self.confusion_matrix = confusion_matrix(
            self.forest, self.data_test, self.target_test
        )
        self.log.append(
            {
                "feature_indices": self.current_feature_indices,
                "confusion_matrix": self.confusion_matrix,
            }
        )
        return self.confusion_matrix

    def predict(self, x):
        if len(x.shape) > 0:
            if x.shape[1] == self.data.nr_features:
                x_tmp = np.take(
                    x.transpose(), self.current_feature_indices, axis=0
                ).transpose()
                return self.forest.predict(x_tmp)
        return self.forest.predict(x)

    def score(self, x):
        x_tmp = x
        if len(x.shape) > 0:
            if x.shape[1] == self.data.nr_features:
                x_tmp = np.take(
                    x.transpose(), self.current_feature_indices, axis=0
                ).transpose()
        return (
            np.array(
                sum(
                    self.forest.estimators_[i].predict(x_tmp)
                    for i in range(self.forest.n_estimators)
                )
            )
            / self.forest.n_estimators
        )

    def save(self, fpath):
        sdir = fpath + "/" + self.__class__.__name__
        os.makedirs(sdir, exist_ok=True)
        dump(self.forest, sdir + "/model.joblib")
        dump(self.log, sdir + "/log.joblib")

    def load(self, fpath):
        sdir = fpath + "/" + self.__class__.__name__
        if os.path.isfile(sdir + "/model.joblib") and os.path.isfile(
            sdir + "/log.joblib"
        ):
            self.log = load(sdir + "/log.joblib")
            feat_indices = np.copy(self.log[-1]["feature_indices"])
            featpos = {
                self.current_feature_indices[i]: i
                for i in range(len(self.current_feature_indices))
            }
            reduced_feats = np.array([featpos[i] for i in feat_indices])
            self.data_train = np.take(
                self.data_train.transpose(), reduced_feats, axis=0
            ).transpose()
            self.data_test = np.take(
                self.data_test.transpose(), reduced_feats, axis=0
            ).transpose()
            self.current_feature_indices = feat_indices
            self.forest = load(sdir + "/model.joblib")
            return True
        else:
            return False

    def plot(self, annotation=None, save_dir=None):
        res = self.score(self.data_test)
        plot_scores(self.data, res, 0.5, self.test_indices, annotation, save_dir)
