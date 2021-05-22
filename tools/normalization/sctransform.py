from tools.basic_tools import RNASeqData
from SCTransform import SCTransform
from anndata import AnnData
from scipy.sparse import csr_matrix


def compute_sctransform(data: RNASeqData):
    assert data.data_array["raw"] is not None
    tmp_csr = AnnData(csr_matrix(data.data_array["raw"]))

    sct_data = SCTransform(
        tmp_csr,
        min_cells=5,
        gmean_eps=1,
        n_genes=2000,
        n_cells=None,  # use all cells
        bin_size=500,
        bw_adjust=3,
        inplace=False,
    )

    data.data_array["sct"] = sct_data.X.toarray()
