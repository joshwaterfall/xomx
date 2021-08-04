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

```python
savedir = os.path.join(os.path.expanduser("~"), "xaiodata", "kidney_classif")
```

## Step 1: Preparing the manifest

We use the 
[GDC Data Transfer Tool](
https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
)
to import data from the Cancer Genome Atlas (TCGA). 
This involves creating a `manifest.txt` file that describes the files to be imported.

The function `gdc_create_manifest()` (`from xaio import gdc_create_manifest`) 
facilitates the creation of this manifest. It is designed to import files of gene 
expression counts performed with [HTSeq](https://github.com/simon-anders/htseq). 

`gdc_create_manifest()` takes in input the disease type (in our case "Adenomas and 
Adenocarcinomas"), the list of project names ("TCGA-KIRC", "TCGA-KIRP", "TCGA-KICH"), 
and the numbers of samples desired for each of these projects (remark: for "TCGA-KICH", 
there are only 66 samples available). It returns a list of Pandas dataframes, one for 
each project.

More information on GDC data can be found on the [GDC Data Portal](
https://portal.gdc.cancer.gov/
).


```python
disease_type = "Adenomas and Adenocarcinomas"
project_list = ["TCGA-KIRC", "TCGA-KIRP", "TCGA-KICH"]
case_numbers = [200, 200, 66]
df_list = gdc_create_manifest(disease_type, project_list, case_numbers)
```

The Pandas library (imported as `pd`) is used to write the concatenation of the
output dataframes to the file `manifest.txt`:

```python
df = pd.concat(df_list)
df.to_csv(
    os.path.join(savedir, "manifest.txt"),
    header=True,
    index=False,
    sep="\t",
    mode="w",
)
```

## Step 2: Importing the data

