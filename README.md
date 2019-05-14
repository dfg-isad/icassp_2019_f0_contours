# Code Repository

## Introduction

This page contains the data & python scripts to reproduce the classification experiments reported in

```
@InProceedings{Abesser:2019:ICASSP,
  author =    {Jakob Abe{\ss}er and Meinard M{\"u}ller},
  title =     {Fundamental Frequency Contour Classification: A Comparison between Hand-Crafted and {CNN}-Based Features},
  booktitle = {Proceedings of the 44th IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
  year =      {2019}
}
```

## Python environment

* In order to set-up the python environment, please install miniconda on your system, then run

```
conda env create -f conda_env.yml
```

## Data preparation

* Download the dataset from [Zenodo](https://zenodo.org/record/2800393#.XNr8A9NKgWo) and copy it into the ```data```
folder

## Run experiments

* Run the experiments using 
```
python main.py
```
* The results will be saved in the ```results``` folder
# 
