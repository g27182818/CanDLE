# CanDLE

This is the code for the Cancer Detection Logistic Engine (CanDLE) method. This is a research effort by the Biomedical Computer Vision gruop (BCV) of Universidad de los Andes authored by Gabriel Mejía and Danniel Moreno.

## Automatically download TCGA data

To download TCGA data first create an R environment in anaconda and install `R`. From the base environment in anaconda we suggest:

```
conda create -n R
conda activate R
conda install -c r r-essentials
```
This will install several needed packages of R in anaconda but also give you `R==3.6.0` version that can generate errors while using R Bioconductor. So, upgrade your version usBiocManager::install("AnnotationHub")ing:

```
conda install -c conda-forge r-base=4.1.0
```
Now you have a fully working R envirment. Now we need to install two packages inside R:
* ***TCGAbiolinks*** ([Bioconductor](https://bioconductor.org/packages/release/bioc/html/TCGAbiolinks.html)): To easily handle the download API of the [GDC project](https://portal.gdc.cancer.gov/).
* ***EDASeq*** ([Bioconductor](https://bioconductor.org/packages/release/bioc/html/EDASeq.html)) To get gene lengths from the ENSEMBLE database.
* ***feather*** ([Docs](https://cran.r-project.org/web/packages/feather/feather.pdf)): To store data in an efficient and fast format that can be read using pandas.
 

To install the do the following in your R environment:

```R
R # This should open an R session
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("BioinformaticsFMRP/TCGAbiolinksGUI.data")
BiocManager::install("AnnotationHub")
BiocManager::install("ExperimentHub")
BiocManager::install("BioinformaticsFMRP/TCGAbiolinks")
BiocManager::install("EDASeq")

install.packages('feather')
```
Notes:
* There is no need to update the `pbdZMQ` when asked by R.
* If the `XML` package gives problems quiting R (`quit()`) and installing this package using `conda install -c conda-forge r-xml` can work. 

Having this you can now quit the R session `quit()` and should be able to run the `tcga_downloader.R` script:
```
Rscript tcga_downloader.R
```
