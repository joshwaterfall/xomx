from xaio_config import output_dir
from tools.basic_tools import RNASeqData

# from IPython import embed as e

data = RNASeqData()
data.save_dir = output_dir + "/dataset/RNASeq/"

data.load(["raw"])
data.compute_std_data()
data.compute_train_and_test_indices_per_annotation()
data.compute_std_values_on_training_sets()
data.compute_std_values_on_training_sets_argsort()

data.raw_data = None
data.save()
