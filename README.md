# Conformal Risk Training: End-to-End Optimization of Conformal Risk Control

[Christopher Yeh](https://chrisyeh96.github.io/), [Nicolas Christianson](https://nicochristianson.com/), [Adam Wierman](https://adamwierman.com/), and [Yisong Yue](https://www.yisongyue.com/)
<br>**California Institute of Technology**, Department of Computing and Mathematical Sciences

This repo contains code for the following paper:

**Conformal Risk Training: End-to-End Optimization of Conformal Risk Control**
<br>C. Yeh, N. Christianson, A. Wierman, Y. Yue
<br>NeurIPS 2025
<br>[**Paper**](INSERT ARXIV LINK)


## Table of Contents

1. [Installation instructions](#installation-instructions)
2. [Example: Tumor segmentation](#example-tumor-segmentation)
3. [Example: Battery storage CVaR control](#example-battery-storage-cvar-control)
4. [Citation](#citation)


## Installation instructions

Code from this repo has been tested on Ubuntu 22.04.

Running code from this repo requires:
- python 3.12
- cvxpy 1.6
- cvxpylayers 0.1.9
- imageio 2.37
- numpy 2.2
- matplotlib 3.10
- pandas 2.2
- pillow 11.1
- pytorch 2.5
- seaborn 0.13
- torchvision 0.20

We recommend using the [conda](https://docs.conda.io/) package manager.

1. Install [miniconda](https://docs.anaconda.com/miniconda/miniconda-install/).

2. Install the packages from the `env.yml` file:
    ```bash
    conda env update --file env.yml --prune
    ```

3. Activate the conda environment
    ```bash
    conda activate e2ecrc
    ```


## Example: Tumor segmentation

1. We use a pretrained [PraNet](https://github.com/DengPingFan/PraNet/) model. Download the weights [here](https://github.com/chrisyeh96/conformal-risk-training/releases/download/v1.0.0/PraNet-19.pth), and save it to `polyps/PraNet-19.pth`.

2. Download the dataset [here](https://github.com/chrisyeh96/conformal-risk-training/releases/download/v1.0.0/polyps_data.zip), and unzip the dataset to `polyps/data`.

   We use a similar dataset as PraNet, with improved pre-processing and filtering of duplicates. See [`polyps/process_data.ipynb`](polyps/process_data.ipynb) for details.

3. For the experiments, run the following commands:
   ```bash
   # post-hoc CRC on pretrained model
   python run_polyp.py crc --alpha 0.01 0.05 0.1 --device cuda

   # train using PraNet cross-entropy loss, then apply post-hoc CRC
   python run_polyp.py trainbase --alpha 0.01 0.05 0.1 --lr 1e-2 1e-3 1e-4 1e-5 1e-6 --device cuda

   # conformal risk training
   python run_polyp.py e2ecrc --alpha 0.01 0.05 0.1 --lr 1e-2 1e-3 1e-4 1e-5 1e-6 --device cuda

   # plot predictions
   python run_polyp.py savepreds -s 0 --tag pretrained --ckpt-path polyps/PraNet-19.pth --device cuda
   python run_polyp.py savepreds -s 0 --tag trainbase --ckpt-path out/polyps/trainbase/lr0.001_s0.pt --device cuda
   python run_polyp.py savepreds -s 0 --tag e2ecrc_a0.01 --ckpt-path out/polyps/e2ecrc/a0.01_lr0.001_s0.pt --device cuda
   python run_polyp.py savepreds -s 0 --tag e2ecrc_a0.05 --ckpt-path out/polyps/e2ecrc/a0.05_lr0.001_s0.pt --device cuda
   python run_polyp.py savepreds -s 0 --tag e2ecrc_a0.10 --ckpt-path out/polyps/e2ecrc/a0.10_lr0.001_s0.pt --device cuda
   ```

4. To generate the figures, run the [`analysis/polyps.ipynb`](analysis/polyps.ipynb) notebook. For the cross-entropy loss table, see [`analysis/polyps_pranet_loss.ipynb`](analysis/polyps_pranet_loss.ipynb).


## Example: Battery storage CVaR control

1. We use the same dataset as in [Donti et al. (2017)](https://github.com/locuslab/e2e-model-learning). The data is already included in this repo under `storage/data`. See [`storage/data/data.ipynb`](storage/data/data.ipynb) for details on how the data was processed.

2. For the experiments, run the following commands:

   ```bash
   # calculate optimal task loss
   python run_storage.py optimal --shuffle

   # pretrain MLP models
   python run_storage.py pretrain --shuffle --device cuda

   # save predictions of MLP models
   python run_storage.py savepreds --shuffle --device cuda

   # post-hoc CRC on pretrained model
   python run_storage.py crc --shuffle --alpha 0 1 2 5 10 15 20 --delta 0.8 0.9 0.95 0.99 --device cuda

   # train using task loss, then apply post-hoc CRC
   python run_storage.py finetune_taskloss --shuffle --alpha 0 1 2 5 10 15 20 --delta 0.8 0.9 0.95 0.99 --lr 1e-2 1e-3 1e-4 1e-5 1e-6 --device cuda

   # E2E CRC
   python run_storage.py e2ecrc --shuffle --alpha 2 5 10 --delta 0.9 0.95 0.99 --lr 1e-2 1e-3 1e-4 1e-5 --device cuda
   ```

3. To generate most figures, run the [`analysis/storage.ipynb`](analysis/storage.ipynb) notebook. For the $t$-hyperparameter sensitivity tables, see [`analysis/storage_vary_t.ipynb`](analysis/storage_vary_t.ipynb). For the plot on varying the size of the calibration set, see [`analysis/storage_vary_calibsize.ipynb`](analysis/storage_vary_calibsize.ipynb).


## Citation

Please cite our papers as follows, or use the BibTeX entry below.

> C. Yeh, N. Christianson, A. Wierman, and Y. Yue, “Conformal Risk Training: End-to-End Optimization of Conformal Risk Control,” in _Advances in Neural Information Processing Systems_, vol. 38, San Diego, CA, USA, Dec. 2025.

```tex
@inproceedings{yeh2025conformal,
  title = {{Conformal Risk Training: End-to-End Optimization of Conformal Risk Control}},
  author = {Yeh, Christopher and Christianson, Nicolas and Wierman, Adam and Yue, Yisong},
  year = 2025,
  month = dec,
  booktitle = {Advances in Neural Information Processing Systems},
  address = {San Diego, CA, USA},
  volume = 38
}
```
