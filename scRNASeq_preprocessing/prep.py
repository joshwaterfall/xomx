from xaio_config import output_dir
from tools.basic_tools import RNASeqData
from scRNASeq_preprocessing.config import (
    scRNASeq_data,
    scRNASeq_features,
    # scRNASeq_barcodes,
)
import csv

# import os
import scipy.io
import numpy as np

mat = scipy.io.mmread(scRNASeq_data)
feature_ids = [
    row[0] + "|" + row[1] for row in csv.reader(open(scRNASeq_features), delimiter="\t")
]

data = RNASeqData()
data.save_dir = output_dir + "/dataset/scRNASeq/"
data.raw_data = mat.todense().transpose()
data.nr_features = len(feature_ids)
data.nr_samples = mat.shape[1]
data.feature_names = np.empty((data.nr_features,), dtype=object)
for i in range(data.nr_features):
    data.feature_names[i] = feature_ids[i]


data.mean_expressions = [np.mean(data.raw_data[:, i]) for i in range(data.nr_features)]
data.std_expressions = [np.std(data.raw_data[:, i]) for i in range(data.nr_features)]
data.std_data = np.copy(data.raw_data)
for i in range(data.nr_features):
    for j in range(data.nr_samples):
        if data.std_expressions[i] == 0.0:
            data.std_data[j, i] = 0.0
        else:
            data.std_data[j, i] = (
                data.std_data[j, i] - data.mean_expressions[i]
            ) / data.std_expressions[i]
data.log_data = np.copy(data.raw_data)
data.epsilon_shift = 1.0
for i in range(data.nr_features):
    data.log_data[:, i] = np.log(data.log_data[:, i] + data.epsilon_shift)
data.maxlog = np.max(data.log_data)
for i in range(data.nr_features):
    data.log_data[:, i] = (data.log_data[:, i] - np.log(data.epsilon_shift)) / (
        data.maxlog - np.log(data.epsilon_shift)
    )
data.feature_shortnames_ref = {}
for i, elt in enumerate(data.feature_names):
    data.feature_shortnames_ref[elt.split("|")[1]] = i
data.save()
