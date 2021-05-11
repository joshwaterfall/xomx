from xaio_config import output_dir
from scRNASeq_preprocessing.config import (
    scRNASeq_data,
    scRNASeq_features,
    # scRNASeq_barcodes,
)
import csv
import os
import scipy.io
import numpy as np


save_dir = output_dir + "/dataset/scRNASeq/"
if not (os.path.exists(save_dir)):
    os.makedirs(save_dir, exist_ok=True)

mat = scipy.io.mmread(scRNASeq_data)
feature_ids = [
    row[0] + "|" + row[1] for row in csv.reader(open(scRNASeq_features), delimiter="\t")
]

data_array = mat.todense()

nr_transcripts = len(feature_ids)
np.save(save_dir + "nr_transcripts.npy", nr_transcripts)
# Load with: np.load(save_dir + 'nr_transcripts.npy', allow_pickle=True).item()
print("(1) " + "saved: " + save_dir + "nr_transcripts.npy")

nr_samples = mat.shape[1]
np.save(save_dir + "nr_samples.npy", nr_samples)
# Load with: np.load(save_dir + 'nr_samples.npy', allow_pickle=True).item()
print("(2) " + "saved: " + save_dir + "nr_samples.npy")

transcripts = np.empty((nr_transcripts,), dtype=object)
for i in range(nr_transcripts):
    transcripts[i] = feature_ids[i]
np.save(save_dir + "transcripts.npy", transcripts)
# Load with: np.load(save_dir + 'transcripts.npy', allow_pickle=True)
print("(3) " + "saved: " + save_dir + "transcripts.npy")

mean_expressions = [np.mean(data_array[i, :]) for i in range(nr_transcripts)]
np.save(save_dir + "mean_expressions.npy", mean_expressions)
# Load with: np.load(save_dir + 'mean_expressions.npy', allow_pickle=True)
print("(4) " + "saved: " + save_dir + "mean_expressions.npy")

std_expressions = [np.std(data_array[i, :]) for i in range(nr_transcripts)]
np.save(save_dir + "std_expressions.npy", std_expressions)
# Load with: np.load(save_dir + 'std_expressions.npy', allow_pickle=True)
print("(5) " + "saved: " + save_dir + "std_expressions.npy")

fp_data = np.memmap(
    save_dir + "raw_data.bin",
    dtype="float32",
    mode="w+",
    shape=(nr_transcripts, nr_samples),
)
fp_data[:] = data_array[:]
del fp_data
# Load with: np.array(np.memmap(save_dir + 'raw_data.bin', dtype='float32', mode='r',
#                     shape=(nr_samples, nr_transcripts)))
print("(6) " + "saved: " + save_dir + "raw_data.bin")

stdmean_data_array = np.copy(data_array)
for i in range(nr_transcripts):
    for j in range(nr_samples):
        if std_expressions[i] == 0.0:
            stdmean_data_array[i, j] = 0.0
        else:
            stdmean_data_array[i, j] = (
                stdmean_data_array[i, j] - mean_expressions[i]
            ) / std_expressions[i]

fp_data = np.memmap(
    save_dir + "data.bin",
    dtype="float32",
    mode="w+",
    shape=(nr_transcripts, nr_samples),
)
fp_data[:] = stdmean_data_array[:]
del fp_data
# Load with: np.array(np.memmap(save_dir + 'data.bin', dtype='float32', mode='r',
#                     shape=(nr_samples, nr_transcripts)))
print("(7) " + "saved: " + save_dir + "data.bin")

epsilon_shift = 1.0
for i in range(nr_transcripts):
    data_array[i, :] = np.log(data_array[i, :] + epsilon_shift)

np.save(save_dir + "epsilon_shift.npy", epsilon_shift)
# Load with: np.load(save_dir + 'epsilon_shift.npy', allow_pickle=True).item()
print("(8) " + "saved: " + save_dir + "epsilon_shift.npy")

maxlog = np.max(data_array)
np.save(save_dir + "maxlog.npy", maxlog)
# Load with: np.load(save_dir + 'maxlog.npy', allow_pickle=True).item()
print("(9) " + "saved: " + save_dir + "maxlog.npy")

for i in range(nr_transcripts):
    data_array[i, :] = (data_array[i, :] - np.log(epsilon_shift)) / (
        maxlog - np.log(epsilon_shift)
    )

fp_data = np.memmap(
    save_dir + "lognorm_data.bin",
    dtype="float32",
    mode="w+",
    shape=(nr_transcripts, nr_samples),
)
fp_data[:] = data_array[:]
del fp_data
# Load with: np.array(np.memmap(save_dir + 'lognorm_data.bin', dtype='float32',
#                     mode='r', shape=(nrows, ncols)))
print("(10) " + "saved: " + save_dir + "lognorm_data.bin")

gene_dict = {}
for i, elt in enumerate(transcripts):
    gene_dict[elt.split("|")[1]] = i
np.save(save_dir + "gene_dict.npy", gene_dict)
# Load with: np.load(save_dir + 'gene_dict.npy', allow_pickle=True).item()
print("(11) " + "saved: " + save_dir + "gene_dict.npy")
