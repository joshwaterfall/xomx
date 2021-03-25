import os

CSV_RNASeq_data = "~/Desktop/work/code/RNAseq/data/big.csv" #TPM-normalized data (transcripts per million: sum of transcripts per sample is 1 million)
CSV_annotations = "~/Desktop/work/code/RNAseq/data/annot_with_fusion_new_names.csv"
CSV_annot_types = "~/Desktop/work/code/RNAseq/data/annot_types.csv"
output_dir = os.path.expanduser("~/Desktop/data/xaio")
data_dir = os.path.expanduser(output_dir + "/dataset/")
xaio_tag = "xaio_tag_1"
