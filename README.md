# CanDLE

Source code for the "CanDLE: Illuminating Biases in Transcriptomic Pan-Cancer Diagnosis" paper presented in the 1st Workshop on Computational Mathematics Modeling in Cancer Analysis ([CMMCA2022](https://cmmca2022.casconf.cn/)) held at MICCAI 2022. You can consult the original video presentation in [this link](https://youtu.be/oL9W5Akdz7w). This is a research effort by the Biomedical Computer Vision group ([BCV](https://biomedicalcomputervision.uniandes.edu.co/)) of Universidad de los Andes authored by Gabriel Mejía, Natasha Bloch and Pablo Arbeláez.

## News

* *18/09/2022:* CanDLE obtained the best paper award prize in the CMMCA2022 workshop!

## Set up

We will first download the required data and then install the needed python dependencies for CanDLE to run. It is important to note that our work approached both the [Vivian et al.](https://www.nature.com/articles/nbt.3772) (Toil) and [Wang et al.](https://www.nature.com/articles/sdata201861) joint TCGA/GTEx datasets. However, Wang data is exclusively used in the `bias_check.py` file. Consequently, if you are only interested in training and testing CanDLE we do not recommend to download the Wang et al dataset.

## Cloning CanDLE repository

```bash
git clone https://github.com/g27182818/CanDLE.git
cd CanDLE
```

## Automatically download [UCSC Toil RNA-seq Recompute data](https://xenabrowser.net/datapages/?cohort=TCGA%20TARGET%20GTEx&removeHub=http%3A%2F%2F127.0.0.1%3A7222)

The `toil_downloader.R` file downloads automatically the principal dataset needed to run CanDLE ([Vivian et al.](https://www.nature.com/articles/nbt.3772)). You can run it with R Studio or with anaconda.

### Using Anaconda

Create an `R` environment:

```bash
conda create --name R
conda activate R
conda install -c conda-forge r-essentials
# conda install -c r r-essentials # Can take some minutes
# conda install -c conda-forge r-base=4.1.0 # Can take some minutes
conda install -c conda-forge r-languageserver
```

Open an R session in terminal to install all needed packages:

```R
R # This should open an R session
install.packages("rlang")
install.packages("UCSCXenaTools")
install.packages('feather')
install.packages("pacman")
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("recount")  # May take several minutes
BiocManager::install("recount3")
BiocManager::install("org.Hs.eg.db")
quit() # This will end the R session you don't need to save workspace (n)
```

Run the downloader:

```bash
Rscript toil_downloader.R
```

Close the R environment:

```bash
conda deactivate
```

### Using R Studio

1. Open an up to date version of R Studio.
2. Install needed packages from console using:

    ```R
    install.packages("rlang")
    install.packages("UCSCXenaTools") # Can take a few minutes
    install.packages('feather')
    ```

3. Run `toil_downloader.R` line by line or the complete script.

The installed packages are:

* ***rlang*** Compatibility reasons.
* ***UCSCXenaTools*** ([Docs](https://cran.r-project.org/web/packages/UCSCXenaTools/UCSCXenaTools.pdf)): To easily handle the download of data from the [UCSCXena Portal](https://xenabrowser.net/datapages/?cohort=TCGA%20TARGET%20GTEx&removeHub=http%3A%2F%2F127.0.0.1%3A7222).
* ***feather*** ([Docs](https://cran.r-project.org/web/packages/feather/feather.pdf)): To store data in an efficient and fast format that can be read using pandas.

Note:

* There can be problems downloading the `TcgaTargetGtex_rsem_gene_tpm.gz` file due to poor internet connection or server problems. When downloaded the final `data/toil_data/data_matrix.feather` file should have a size of 8.62 Gb. If it does not have this size, it is recommended to download the file directly from [this link](https://toil.xenahubs.net/download/TcgaTargetGtex_rsem_gene_tpm.gz) to the `data/toil_download/` folder and then re-run `toil_downloader.R`.
* There can be problems downloading the `TcgaTargetGtex_gene_expected_count.gz` file due to poor internet connection or server problems. When downloaded the final `data/toil_data/count_matrix.feather` file should have a size of 8.61 Gb. If it does not have this size, it is recommended to download the file directly from [this link](https://toil.xenahubs.net/download/TcgaTargetGtex_gene_expected_count.gz) to the `data/toil_download/` folder and then re-run `toil_downloader.R`.

## Automatically download [Recount3](http://rna.recount.bio/) data

As all the needed packages are installed in the `R` environment, we can run the `recount_downloader.R` file to download the data.

```bash
Rscript recount3_downloader.R
```

## Get gene name mapping

```bash
Rscript gene_names.R
```

## Automatically download [Wang et al.](https://www.nature.com/articles/sdata201861) joint TCGA/GTEx dataset

This database is hosted in [this](https://doi.org/10.6084/m9.figshare.5330593) figshare link but we will download it programmatically.

1. First make a directory to host the data:

   ```bash
   mkdir -p data
   cd data
   mkdir -p wang_data
   cd wang_data
   ```

2. Download data with wget:

   ```bash
   wget -O raw_data.zip https://figshare.com/ndownloader/articles/5330539/versions/2
   cd ..
   cd ..
   ```

This raw data file will be uncompressed and read by the `WangDataset()` class in the `datasets.py` file.

## Download required python dependencies for CanDLE

To install all required dependencies run each line sequentially:

```bash
conda create -n candle
conda activate candle
conda install pytorch torchvision cudatoolkit=9.0 python=3.9 -c pytorch
pip install matplotlib
pip install seaborn
pip install pyarrow
pip install adjustText
pip install sklearn
pip install tqdm
pip install scanpy
pip install qnorm
pip install combat
pip install thundersvm
```

**Note:**
Only the `bias_check.py` and `sota_detection.py` files train predictive models in CPU by default. All the other files (`main.py`, `all_vs_one_exp.py`, `interpretation.py`, `sota_classification.py`) are programed to run in GPU hardware.

## Running the bias check results

The `bias_check.py` file takes one dataset (`toil`, `toil_norm` or `wang`, parameter specified internally) and trains a linear support vector machine to classify the data source (TCGA/GTEx). It prints the classification results to terminal and produces separation histograms of the distance to the final SVM hyperplane (stored in the `Figures/` directory).

```bash
python bias_check.py
```

This code runs in CPU and can take significant time to run (minutes-hours) with the normalized toil dataset.

## Running a single CanDLE classification experiment

You can train and test CanDLE and change its main parameters using the `one_exp.sh` file. First change it to executable:

```bash
chmod u+x one_exp.sh
```

And now you can prove CanDLE running:

```bash
./one_exp.sh
```

You can run multiple experiments by changing the internal parameters of `one_exp.sh`. The gene ranking generated by these single CanDLE model is stored in `Rankings/1_candle_ranking.csv` and a plot of the weights of a random cancer class is stored in `Figures/random_class_weights_plot.png`

## Running all-vs-one detection experiments

The `all_vs_one_exp.py` file trains serially 33 CanDLE models each one to detect a single cancer type and stores the results in the `Results/CanDLE_all_vs_one_exp_1_epoch` directory. There are some internal parameters in the file that can be changed if desired.

```bash
python all_vs_one_exp.py
```

## Running CanDLE interpretation

The `interpretation.py` file trains serially 100 pan-cancer classification CanDLE models and stores each model's results in the `Results/interpretation` directory. Each time, the method is trained in a different random partition of the data. After that, we use bootstrapping to perform a Wald Z test in the significance of CanDLE weights and obtain an importance ranking (stored in `Rankings/100_candle_ranking.csv`) There are some internal parameters in the file that can be changed if desired.

```bash
python interpretation.py
```

## Running state of the art methods

1. A re-implementation of the [Hong et al.](https://www.nature.com/articles/s41598-022-13665-5) classification method can be trained using:

   ```bash
   python sota_classification.py
   ```

   This code displays the results in terminal and does not save logs.

2. An adaptation of the [Quinn et al.](https://www.frontiersin.org/articles/10.3389/fgene.2019.00599/full) source [code](https://github.com/thinng/tissue_detector) specifically tailored to cancer detection in our framework can be run using the following:

   ```bash
   python sota_detection.py
   ```

   This code displays results in terminal, does not save logs and stores the original per-class detection visualization figures of Quinn's work in the `Figures/sota_detection` directory.
