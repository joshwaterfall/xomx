import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

# from IPython import embed as e


class RFEExtraTrees:
    def __init__(self, data, annotation):
        self.data = data
        self.annotation = annotation
        self.init_selection_size = 4000
        self.current_feature_indices = (
            self.data.expressions_on_training_sets_argsort[annotation][
                : (self.init_selection_size // 2)
            ].tolist()
            + self.data.expressions_on_training_sets_argsort[annotation][
                -(self.init_selection_size - self.init_selection_size // 2) :
            ].tolist()
        )
        self.train_indices = sum(self.data.annot_index_train.values(), [])
        self.test_indices = sum(self.data.annot_index_test.values(), [])
        self.data_train = np.take(
            np.take(self.data.data.transpose(), self.current_feature_indices, axis=0),
            self.train_indices,
            axis=1,
        ).transpose()
        self.target_train = np.zeros(self.data.nr_samples)
        self.target_train[self.data.annot_index_train[self.annotation]] = 1.0
        self.target_train = np.take(self.target_train, self.train_indices, axis=0)
        self.data_test = np.take(
            np.take(self.data.data.transpose(), self.current_feature_indices, axis=0),
            self.test_indices,
            axis=1,
        ).transpose()
        self.target_test = np.zeros(self.data.nr_samples)
        self.target_test[self.data.annot_index_test[self.annotation]] = 1.0
        self.target_test = np.take(self.target_test, self.test_indices, axis=0)

        # self.genes_progress = []
        # self.features_progress = []
        # self.mcc_progress = []

    def select_features(self, n):
        assert n <= self.data_train.shape[1]
        forest = ExtraTreesClassifier(n_estimators=450, random_state=0)
        forest.fit(self.data_train, self.target_train)
        sorted_feats = np.argsort(forest.feature_importances_)[::-1]
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

    # def fit(self):
    #     forest = ExtraTreesClassifier(n_estimators=450, random_state=0)
    #     forest.fit(self.data, self.y)
    #     mcc, mccstr = mcc_test(forest, self.data_test, self.y_test)
    #     print(mccstr)
    #
    #     self.genes_progress.append(self.true_indices)
    #     self.features_progress.append(len(self.true_indices))
    #     self.mcc_progress.append(mcc)
    #
    #     # importances = forest.feature_importances_
    #     # forest_indices = np.argsort(importances)[::-1]
    #     # reduced_indices = list(forest_indices[:30])
    #
    #     for siz in [4000, 100, 30, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
    #     # for siz in [5, 4, 3, 2, 1]:
    #         sorted_feats = np.argsort(forest.feature_importances_)[::-1]
    #         reduced_feats = list(sorted_feats[:siz])
    #
    #         self.true_indices = np.take(self.true_indices, reduced_feats, axis=0)
    #
    #         data_processed = np.take(self.data.transpose(), reduced_feats, axis=0)
    #         .transpose()
    #         data_test_processed = np.take(self.data_test.transpose(), reduced_feats
    #         , axis=0).transpose()
    #
    #         forest = ExtraTreesClassifier(n_estimators=450, random_state=0)
    #         forest.fit(data_processed, self.y)
    #         mcc, mccstr = mcc_test(forest, data_test_processed, self.y_test)
    #
    #         print(mccstr)
    #
    #         self.genes_progress.append(self.true_indices)
    #         self.features_progress.append(len(self.true_indices))
    #         self.mcc_progress.append(mcc)
