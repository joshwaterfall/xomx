# XAIO - Biomarker Discovery Tutorial

-----

The objective of this tutorial is to use a recursive feature elimination method on 
RNA-Seq data to identify gene biomarkers for the differential diagnosis of three 
types of kidney cancer: Kidney Renal Clear Cell Carcinoma (KIRC), Kidney Renal 
Papillary Cell Carcinoma (KIRP), and Kidney Renal Clear Cell Carcinoma (KICH).

**Repeated executions of the `kidney_classif.py` file perform each of the 7 steps of 
the tutorial, one by one.**

After the imports, the following line defines the folder (`~/xaiodata/kidney_classif`) 
in which data and outputs will be stored:

`savedir = os.path.join(os.path.expanduser("~"), "xaiodata", "kidney_classif")`


