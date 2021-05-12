from xaio_config import output_dir
from tools.basic_tools import RNASeqData

# from IPython import embed as e

data = RNASeqData()
data.save_dir = output_dir + "/dataset/RNASeq/"

data.load(["raw"])
data.compute_log_data()
data.raw_data = None
data.save()
