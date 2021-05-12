from xaio_config import output_dir
from tools.basic_tools import RNASeqData
from RNASeq_preprocessing.config import (
    CSV_RNASeq_data,
    CSV_annotations,
    CSV_annot_types,
)
import pandas as pd
import numpy as np
from IPython import embed as e

data = RNASeqData()
data.save_dir = output_dir + "/dataset/RNASeq/"

rnaseq_read = pd.read_table(CSV_RNASeq_data, sep=",", header=0, engine="c", nrows=1)
data.sample_ids = rnaseq_read.columns.values[1:]
annotations_dict = dict(pd.read_csv(CSV_annotations, sep=",").to_numpy())
data.sample_annotations = np.empty_like(data.sample_ids)
for i, s_id in enumerate(data.sample_ids):
    data.sample_annotations[i] = annotations_dict[s_id]
data.compute_sample_indices()
data.compute_all_annotations()
data.compute_sample_indices_per_annotation()
origins_dict = dict(pd.read_csv(CSV_annot_types, sep=",").to_numpy())
data.sample_origins = np.empty_like(data.sample_ids)
for i, s_id in enumerate(data.sample_ids):
    data.sample_origins[i] = origins_dict[s_id]
data.compute_sample_origins_per_annotation()

rnaseq_array = pd.read_table(CSV_RNASeq_data, header=0, engine="c").to_numpy()
data.nr_features = rnaseq_array.shape[0]
data.nr_samples = len(rnaseq_array[0][0].split(",")) - 1
raw_data_transpose = np.zeros((data.nr_features, data.nr_samples))
data.feature_names = np.empty((data.nr_samples,), dtype=object)
for i in range(data.nr_samples):
    row_value = rnaseq_array[i][0].split(",")
    data.feature_names[i] = row_value[0]
    raw_data_transpose[i, :] = row_value[1:]
    if not i % (data.nr_samples // 100):
        print(i // (data.nr_samples // 100), "%\r", end="")

data.raw_data = raw_data_transpose.transpose()
data.compute_mean_expressions()
data.compute_std_expressions()
data.compute_feature_shortnames_ref()

data.save()

e()
