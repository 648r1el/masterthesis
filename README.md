# Repository of my Masterthesis 

This repository contains the methods used for my Masterthesis. If you want to use some methods from this repository or were inspired by it, you can cite the thesis as referenced below

## Structure of this repository

This repository contains three different parts which were applied in the following order

1. data_download: This directory contains a jupyter notebook used to download the TCGA data from UCSC used in this work. __It needs two more files to be fully executed__: the _methylation file from the Illumina 450k
BeadChip_ and the _most recent gencode annotation file_ (version 49 of gencode was used in this work, you can update it when a new one is published). The file prepare_multiomics_data.py can be used if the data is retrieved
directly from GDC via the TCGABiolinks package in R.

2. MOFA2: Contains the scripts used to run MOFA for the thesis. The used data has to be stored in a directory called Input and the trained models are stored in a directory Trained_models.

3. OmiEmbed: The customized version of OmiEmbed, where the fourth omic layer lncRNA is implemented.

## Usage of each directory

### Download the needed data

The jupyter notebook download_UCSC_tcga_data.ipynb can be executed as soon as the two previously mentioned files are added to the directory. The usage of this notebook is straightforward.

### MOFA2

The omics files used to train MOFA have to be stored in a directory called Input and must have the form mofa_{TCGA_Project}_{omic_layer}.tsv(.bz2)?; only if your omic file is too large to process, it should be given in
the bz2 format. The successfully trained model is stored in a directory called Trained_models. Currently, the script is _restricted to accept exactly 4 omics_, you can add further omics in a hard coded way (or commit a
dynamic approach ;P).

If patients shall be grouped by certain clinical criterion (e.g. by cancer stage), a group file with all patient ids and a column with the wanted information must be provided; two examples are given with the 
mofa_coad_groups.tsv and mofa_read_groups.tsv. _The script does also run without this file_, the feature might still be beneficial in some cases.

The downstream analysis can be executed with the jupyter notebook mofa2_visualize_output.ipynb. It will _create a directory Jupyter_outputs_ where all graphics produced by the notebook are stored.

### OmiEmbed

In its core, OmiEmbed still works like its original publication. Some input flags were added for more convenience and more flexible usage of the tool. In addition to the standard version, further scripts have been added:

- grid_search.sh: A bash script to execute a grid search to find the optimal learning rate and batch size parameter. It accepts the number of epochs as input flags to adjust them for a better performance.
The results are stored in a directory called grid_search_hyperparameters for the training and the test performance metrics, respectively

- run_experiments.sh: A bash script to execute the experiments of the thesis (COAD Normal classification and All Cancer Classification). This script can be used as a guide line how an experiment can be executed
multiple times

- explain_model.py: A python script to explain the trained models with SHAP values. This script executes the interpretation for _each run in the checkpoints directory belonging to the experiment_ and stores the calculated
SHAP values in the directory explanation. In the explanation directory, subdirectories for each omic layer are created when the explain_model.py script is executed.

- explanation directory: The explanation directory contains a jupyter notebook to evaluate the performance of all runs of an experiment and to plot the SHAP values. The part where the the training and test performance is
evaluated is computationally unefficient and needs an update, but still works decently for a single cancer class

The mentioned files correspond to the recommended order of executing this version of OmiEmbed as follows:

1. Run a grid search to find the optimal set of the parameters learning rate, batch size, and number of epochs
2. Execute the experiments and run it as often as desired
3. Calculate the SHAP values for each experiment run
4. Evaluate the results with the downstream analysis

## 📚 Citation

If you use this repository in your research, please cite:

> Abrantes, Gabriel (2026).  
> *Identification of epigenetic biomarkers in colorectal cancer with Variational Bayes Methods*. Master’s thesis, Universität Bielefeld in collaboration with Insituto Superior Técnico, Universiaded de Lisboa.  
