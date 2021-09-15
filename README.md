# XAIO

XAIO is a python library for eXplainable AI in Oncogenomics.

-----

Recommended installation steps (with conda): 
```
git clone git@github.com:perrin-isir/xaio.git
cd xaio
conda env create -f environment.yaml
conda activate xaiov
```
Then, use the following command to install the xaio library within the xaiov virtual
environment: 
```
pip install -e .
```
-----
Tutorials (in [xaio/tutorials/](xaio/tutorials/)) are the best way to learn to use
the XAIO library.

Here is the list of tutorials:
* [kidney_classif.md](xaio/tutorials/kidney_classif.md) (goal:  use a recursive feature 
elimination method on RNA-Seq data to identify gene biomarkers for the differential 
diagnosis of three types of kidney cancer)