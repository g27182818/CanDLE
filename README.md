# CanDLE

Source code for the Cancer Detection Logistic Engine (CanDLE) method. This is a research effort by the Biomedical Computer Vision gruop (BCV) of Universidad de los Andes authored by Gabriel Mejía.

## Automatically download UCSC Toil RNA-seq Recompute data

This is done using the `toil_downloader.R` file. You can run it with R Studio or with anaconda.

### Using Anaconda:
Create an `R` enviroment:
```
conda create --name R
conda activate R
conda install -c r r-essentials
conda install -c conda-forge r-base=4.1.0
conda install -c conda-forge r-languageserver
```
Open an R session to install needer packages:
```R
R # This should open an R session
install.packages("UCSCXenaTools") # Can take a few minutes
install.packages('feather')
quit() # This will end the R session
```
Run the downloader:
```
Rscript toil_downloader.R
```

### Using R Studio:
1. Open an up to date version of R Studio.
2. Install needed packages from console using:
```R
install.packages("UCSCXenaTools") # Can take a few minutes
install.packages('feather')
```
3. Run `toil_downloader.R` line by line or the complete script.


The installed packages are:
* ***UCSCXenaTools*** ([Docs](https://cran.r-project.org/web/packages/UCSCXenaTools/UCSCXenaTools.pdf)): To easily handle the download of data from the [UCSCXena Portal](https://xenabrowser.net/datapages/?cohort=TCGA%20TARGET%20GTEx&removeHub=http%3A%2F%2F127.0.0.1%3A7222).
* ***feather*** ([Docs](https://cran.r-project.org/web/packages/feather/feather.pdf)): To store data in an efficient and fast format that can be read using pandas.

Note:
* There can be problems downloading the `TcgaTargetGtex_rsem_gene_tpm.gz` file due to poor internet connection or server problems. When downloaded the final `data/toil_data/data_matrix.feather` file should have a size of 8.52 Gb. If it does not have this size, it is recomended to download directly the file from [this link](https://toil.xenahubs.net/download/TcgaTargetGtex_rsem_gene_tpm.gz) to the `data/toil_download/` folder and then re-run `toil_downloader.R`.

## Download required python dependencies for CanDLE

To install all requiered dependencies run each line sequentially:

```bash
conda create --name candle
conda activate candle
conda install -y pytorch=1.9.0 torchvision cudatoolkit=10.2 python=3.7 -c pytorch
conda install pyg -c pyg
pip install matplotlib
pip install seaborn
pip install pyarrow
pip install adjustText
```
To finalize the setup run the `make mappers.py` file to save important files to create `ToilDataset` objects:

```
python make_mappers.py
```

## Running a single experiment

First change the mode of the `one_exp.sh` file to excecutable:
```
chmod u+x one_exp.sh
```

And now you can train CanDLE running:
```
./one_exp.sh
```

You can run multiple experiments by changing the internal parameters of `one_exp.sh`

