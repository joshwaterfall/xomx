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

The `gdc_create_manifest()` function (`from xaio import gdc_create_manifest`) 
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

Once the manifest is written, individual samples are downloaded to a temporary folder
(`tmpdir_GDCsamples/`) with the following command:

`gdc-client download -d tmpdir_GDCsamples -m /path/to/manifest.txt`

This requires the `gdc-client`, which can be downloaded at: 
https://gdc.cancer.gov/access-data/gdc-data-transfer-tool

## Step 3: Creating and saving the XAIOData object

First, we use the `gdc_create_data_matrix()` function...

## Step 4: Annotating the samples

## Step 5: Basic pre-processing

## Step 6: Training binary classifiers and performing recursive feature elimination

```python
nr_annotations = len(xdata.all_annotations)
feature_selector = np.empty(nr_annotations, dtype=object)
for i in range(nr_annotations):
    print("Annotation: " + xdata.all_annotations[i])
    feature_selector[i] = RFEExtraTrees(
        xdata,
        xdata.all_annotations[i],
        init_selection_size=4000,
        n_estimators=450,
        random_state=0,
    )
    feature_selector[i].init()
    for siz in [100, 30, 20, 15, 10]:
        print("Selecting", siz, "features...")
        feature_selector[i].select_features(siz)
    feature_selector[i].save(
        os.path.join(
            savedir, "xdata_small", "feature_selectors", xdata.all_annotations[i]
        )
    )
```

## Step 7: Visualizing results

+ Standard deviation vs. mean value for all features:

```python
xdata.function_scatter(
    lambda idx: xdata.feature_mean_values[idx],
    lambda idx: xdata.feature_standard_deviations[idx],
    "features")
```
![alt text](imgs/tuto1_mean_vs_std_deviation.png 
"Standard deviation vs. mean value for all features")

+ Scores on the test dataset for the "TCGA-KIRC" binary classifier 
(positive samples are above the y=0.5 line):
```python
feature_selector[0].plot()
```
![alt text](imgs/tuto1_KIRC_scores.png 
"Scores on the test dataset for the 'TCGA-KIRC' binary classifier")


+ 2D UMAP projection of the log-normalized data limited to the 30 selected features
(10 for each type of cancer):

```python
xdata.umap_plot("log")
```
![alt text](imgs/tuto1_UMAP.png 
"2D UMAP plot")

This plot suggests a division into 4 categories of samples rather than 3. A possible 
interpretation is that the samples in the bottom left cluster may have been 
miscategorized, and are in fact not cases of KIRC, KIRP or KICH.

+ Log-normalized values accross all samples, for the 30 genes that have been 
selected:
```python
xdata.feature_plot(gene_list, "log")
```

![alt text](imgs/tuto1_30features.png 
"Log-normalized values accross all samples for the 30 selected features")

For the first gene at the top, ENSG00000185633.9, the differential expression 
(KIRC vs. other samples) seems particularly pronounced.

+ Read counts for ENSG00000185633.9 accross all samples:
```python
xdata.feature_plot("ENSG00000185633.9", "raw")
```
![alt text](imgs/tuto1_NDUFA4L2_KIRC.png 
"Read counts for ENSG00000185633.9")

![alt text](imgs/tuto1_ELF3_KIRP.png 
"Read counts for ENSG00000163435.14")

![alt text](imgs/tuto1_FOXI1_KICH.png 
"Read counts for ENSG00000168269.8")

It turns out that ENSG00000185633.9 is the gene "NDUFA4L2", which is known to be a 
biomarker of KIRC. See the following publication:

D. R. Minton et al., *Role of NADH Dehydrogenase (Ubiquinone) 1 alpha subcomplex 4-like 2 in clear 
cell renal cell carcinoma*, 
Clin Cancer Res. 2016 Jun 1;22(11):2791-801. doi: [10.1158/1078-0432.CCR-15-1511](
https://doi.org/10.1158/1078-0432.CCR-15-1511
)
