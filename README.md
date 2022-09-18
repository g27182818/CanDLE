# CanDLE

Source code for the Cancer Diagnosis Logistic Engine (CanDLE) method. This is a research effort by the Biomedical Computer Vision gruop (BCV) of Universidad de los Andes authored by Gabriel Mejía, Natasha Bloch and Pablo Arbelaez.

## Set up

We will first download the required data and then construct mappers for datasets objects.

## Automatically download UCSC Toil RNA-seq Recompute data

The `toil_downloader.R` file downloads automatically the principal dataset needed to un CanDLE ([Vivian et al.](https://www.nature.com/articles/nbt.3772)). You can run it with R Studio or with anaconda.

### Using Anaconda

Create an `R` enviroment:

```bash
conda create --name R
conda activate R
conda install -c r r-essentials
conda install -c conda-forge r-base=4.1.0
conda install -c conda-forge r-languageserver
```

Open an R session in terminal to install all needed packages:

```R
R # This should open an R session
install.packages("UCSCXenaTools") # Can take a few minutes
install.packages('feather')
quit() # This will end the R session you dont need to save workspace (n)
```

Run the downloader:

```bash
Rscript toil_downloader.R
```

Close the R enviroment:

```bash
conda deactivate
```

### Using R Studio

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

* There can be problems downloading the `TcgaTargetGtex_rsem_gene_tpm.gz` file due to poor internet connection or server problems. When downloaded the final `data/toil_data/data_matrix.feather` file should have a size of 8.62 Gb. If it does not have this size, it is recomended to download the file directly from [this link](https://toil.xenahubs.net/download/TcgaTargetGtex_rsem_gene_tpm.gz) to the `data/toil_download/` folder and then re-run `toil_downloader.R`.
* There can be problems downloading the `TcgaTargetGtex_gene_expected_count.gz` file due to poor internet connection or server problems. When downloaded the final `data/toil_data/count_matrix.feather` file should have a size of 8.61 Gb. If it does not have this size, it is recomended to download the file directly from [this link](https://toil.xenahubs.net/download/TcgaTargetGtex_gene_expected_count.gz) to the `data/toil_download/` folder and then re-run `toil_downloader.R`.

## Download required python dependencies for CanDLE

To install all requiered dependencies run each line sequentially:

```bash
conda create -n candle
conda activate candle
conda install pytorch torchvision cudatoolkit=10.2 python=3.9 -c pytorch
pip install matplotlib
pip install seaborn
pip install pyarrow
pip install networkx
pip install adjustText
pip install sklearn
pip install tqdm
```

To finalize the setup run the `make mappers.py` file to save important files to create `ToilDataset` objects:

```bash
python make_mappers.py
```

## Running a single classification experiment

You can train and test CanDLE and change its main parameters using the `one_exp.sh` file. First change the mode to excecutable:

```bash
chmod u+x one_exp.sh
```

And now you can prove CanDLE running:

```bash
./one_exp.sh
```

You can run multiple experiments by changing the internal parameters of `one_exp.sh`
