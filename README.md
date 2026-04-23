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
