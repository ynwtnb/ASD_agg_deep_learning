# ASD_agg_deep_learning

This is a repository for analyzing the ASD aggression dataset using three deep learning models: ShapeNet, PatchTST, and Temporal Convolutional Network.

## Demo

| Notebook | Open |
|---|---|
| Model loading and inference Demo | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11ITo-ZsbqYSTk1PzAUHWRHcm6T_NuIaw#scrollTo=b423fe66) |

**Assets:** Models and data snippets are downloaded automatically from [this Google Drive folder](https://drive.google.com/drive/folders/1bNNxU7DAms4sxym3HIgOtyojjCAaZsp2).

## Structure

```
project/
├── README.md
├── .gitignore
├── environment.yml               # shared dependencies
│
├── data/
│
├── models/
│   ├── shapenet/
│   │   ├── README.md
│   ├── patchtst/
│   │   ├── README.md
│   └── tcn/
│       ├── README.md
│
├── reference/                    # reference code from past projects
│
├── shared/                       # shared utility codes 
│
├── experiments/
│   └── results/
│       ├── shapenet/
│       ├── patchtst/
│       └── tcn/
│
├── notebooks/                    # notebooks for analyzing the results
│
└── scripts/                      # shell scripts
```
<<<<<<< HEAD

## Set up

1. Clone this repo.

```bash
git clone https://github.com/ynwtnb/ASD_agg_deep_learning.git
```

2. Request a node on the short partition with 1 CPU core. Then, load the anaconda module.

```bash
srun --partition=short --nodes=1 --cpus-per-task=1 --pty /bin/bash
module load miniconda3/25.9.1
```

3. Change the `prefix` of the `environment.yml` file to the path where you want to save the environment.

4. Run `scripts/setup.sh` to set up the environment for the first time.


=======
>>>>>>> fc94055e7224e68ecbefc07c17fd87f4a06ef64e
