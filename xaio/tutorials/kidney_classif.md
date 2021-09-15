# XAIO - Biomarker Discovery Tutorial

-----

The objective of this tutorial is to use a recursive feature elimination method on 
RNA-Seq data to identify gene biomarkers for the differential diagnosis of three 
types of kidney cancer: Kidney Renal Clear Cell Carcinoma (KIRC), Kidney Renal 
Papillary Cell Carcinoma (KIRP), and Kidney Renal Clear Cell Carcinoma (KICH).

**Repeated executions of the `kidney_classif.py` file perform each of the 7 steps of 
the tutorial, one by one.** A specific step can also be chosen using an integer
argument. For instance, `python kidney_classif.py 1` executes the step 1.

-----

After the imports, the following lines define the folder 
in which data and outputs will be stored:
```python
args = get_args()
savedir = args.savedir
```
By default, `savedir` is `~/results/xaio/kidney_classif`, but it can be modified using a 
`--savedir` argument in input (e.g. `python kidney_classif.py --savedir /tmp`).


## Step 1: Preparing the manifest

We use the 
[GDC Data Transfer Tool](
https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
)
to import data from the Cancer Genome Atlas (TCGA). 
This involves creating a `manifest.txt` file that describes the files to be imported.

The `gdc_create_manifest()` function (`from xaio import gdc_create_manifest`) 
facilitates the creation of this manifest. It is designed to import files of gene 
expression counts obtained with [HTSeq](https://github.com/simon-anders/htseq). 

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

In linux, the command `export PATH=$PATH:/path/to/gdc-client` can be useful to make
sure that the `gdc-client` is found during the execution of `kidney_classif.py`.

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
    "features",
    xlog_scale=True,
    ylog_scale=True,
)
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
xdata.reduce_features(gene_list)
xdata.compute_normalization("log")
xdata.umap_plot("log")
```
![alt text](imgs/tuto1_UMAP.png 
"2D UMAP plot")

We observe 3 distinct clusters corresponding to the three categories
KIRC, KIRP and KICH. Remark: it may be possible that some of the 
samples have been miscategorized.

+ Log-normalized values accross all samples, for the 30 genes that have been 
selected:
```python
xdata.feature_plot(gene_list, "log")
```

![alt text](imgs/tuto1_30features.png 
"Log-normalized values accross all samples for the 30 selected features")

The recursive feature elimination procedure returned 30 features whose combined values 
allow us to distinguish the 3 categories of cancers. A strong contrast can also be 
observed for some individual features. For example, in the figure above, 
the features ENSG00000185633.9 (for KIRC), ENSG00000168269.8 (for KICH) and
ENSG00000163435.14 (for KIRP) stand out.

Let us plot the read counts accross all samples for each of these 3 features.

+ ENSG00000185633.9 (NDUFA4L2 gene):
```python
xdata.feature_plot("ENSG00000185633.9", "raw")
```
![alt text](imgs/tuto1_NDUFA4L2_KIRC.png 
"Read counts for ENSG00000185633.9")

+ ENSG00000163435.14 (ELF3 gene):
```python
xdata.feature_plot("ENSG00000163435.14", "raw")
```
![alt text](imgs/tuto1_ELF3_KIRP.png 
"Read counts for ENSG00000163435.14")

+ ENSG00000168269.8 (FOXI1 gene):
```python
xdata.feature_plot("ENSG00000168269.8", "raw")
```
![alt text](imgs/tuto1_FOXI1_KICH.png 
"Read counts for ENSG00000168269.8")

Studies on the role of these genes in kidney cancers can be found in the literature:
+ In the following publication, the gene NDUFA4L2 (ENSG00000185633.9) is analyzed as a 
biomarker for KIRC:
[D. R. Minton et al., *Role of NADH Dehydrogenase (Ubiquinone) 1 alpha subcomplex 4-like 
2 in clear cell renal cell carcinoma*, 
Clin Cancer Res. 2016 Jun 1;22(11):2791-801. doi: [10.1158/1078-0432.CCR-15-1511](
https://doi.org/10.1158/1078-0432.CCR-15-1511
)].
+ In [A. O. Osunkoya et al., *Diagnostic biomarkers for renal cell carcinoma: selection 
using novel bioinformatics systems for microarray data analysis*, 
Hum Pathol. 2009 Dec; 40(12): 1671–1678. doi: [10.1016/j.humpath.2009.05.006](
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2783948/
)], the gene ELF3 (ENSG00000163435.14) is verified as a biomarker for KIRP.
+ Finally, [D. Lindgren et al., *Cell-Type-Specific Gene Programs of the Normal Human 
Nephron Define Kidney Cancer Subtypes*, Cell Reports 2017 Aug; 20(6): 1476-1489. 
doi: [10.1016/j.celrep.2017.07.043](
https://doi.org/10.1016/j.celrep.2017.07.043
)] identifies the transcription factor FOXI1 (ENSG00000168269.8) to be drastically 
overexpressed in KICH.