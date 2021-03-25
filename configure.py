CSV_RNASeq_data = input(
    "Input path for CSV_RNASeq_data (see description in xaio_config.py):"
)
CSV_annotations = input(
    "Input path for CSV_annotations (see description in xaio_config.py):"
)
CSV_annot_types = input(
    "Input path for CSV_annot_types (see description in xaio_config.py):"
)
output_dir = input("Input path for output_dir (see description in xaio_config.py):")

with open("config.txt", "w") as f:
    f.write(CSV_RNASeq_data + "\n")
    f.write(CSV_annotations + "\n")
    f.write(CSV_annot_types + "\n")
    f.write(output_dir + "\n")
