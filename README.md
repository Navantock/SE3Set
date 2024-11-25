<div align="center">

# SE3Set: Harnessing equivariant hypergraph neural networks for molecular representation learning

</div>

<br>

## 📌  Introduction

SE3Set is an equivariant hypergraph neural network.

<br>


## 🚀  Installation
No installation is required. It is recommended to use gitclone to fetch the code directly and install the following dependencies to configure the environment.

```bash
# clone project
git clone https://github.com/Navantock/SE3Set.git
cd SE3Set

# [OPTIONAL][RECOMMEND] create conda environment
[Optional] mamba create -n SE3Set python==3.10
[Optional] mamba activate SE3Set

# [MANDATORY] Install dependencies
# [RECOMMEND] Install from a new environment
# Take CUDA==12.4 version as an example
mamba install -c conda-forge openff-toolkit 
pip install networkx[default]
pip install PyYAML
pip install rdkit
pip install ase
mamba install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
pip install lightning==2.4
pip install timm==0.4.12
pip install e3nn
```

## ✨   Usage 

To start training, you need to generate a preprocessed torch_geometric dataset, and determine some dataset-based parameters. Then you can run train.py to train your SE3Set model. For QM9, MD17, MD22 datasets experiments in our paper, we have provided a python script to preprocess these datasets. 

See subsections under this section for detailed step-by-step instructions.

### Prepare Configs

The training config file mainly contains of three parts: data parameters, model parameters, training parameters.  The template are in <u>./train_config</u>. We would like to give a brief description for the important parameters of dataset as these parameters are very closely tied to our fragment method.

- `data_params`: the parameters to generate the dataset. 
  - `root`: the torch_geometric dataset root.
  - `label`: the training target. For MD17&MD22 dataset, the label refers to the type of the molecule system.
  - `frac_list`: the train/valid/test split ratio/number.
  - `fragmentor_kwargs`: some fragmentor arguments.  For detailed information please see our article and code.
    - `empr_bo_method`: the method to calculate empirical bond order.
    - `topoBO_threshold`: threshold to determine either to break the bond or not. 
    - `min_kernel_atoms_num`: the minimum atoms number in one fragment.
    - `max_kernel_atoms_num`: the maximum atoms number in one fragment. It can control the fragment size, but it is not absolutely fulfilling.
    - `min_bo_threshold`: the minimum bond order valur to determine whether the fragment can have less atoms than `min_kernel_atoms_num`.
    - `expand_fbo_threshold`: the $c_w$ for explicit overlap. Details see the article.
    - `overlap_level`: the level that overlaps generated. "groups" is used for our experiments. If you do not set this argument, the fragment will not intersect with each other.
  - `he_type`: implicit or explicit overlap method. **Please set "implicit" if you do not use a overlap in fragment_kwargs**
  - `implicit_rc`: the $r_c$ for implicit overlap method.
- `model_params`: the endowment parameters of the model, including the dimensions of each module and some choice of methods.
- `train_params`: hyperparameters used for training and control over scheduler, training logs, etc.

If you want to use our parameters to get a quick start for the experiments in our article, there is no need to change many parameters. You can simply use our template for different datasets and change the parameters below

- `dataset_name`
- `data_params`
  - `root`
  - `label`
  - `frac_list`
- `train_params`
  - `save_path`

### Prepare Dataset

Our model requires some pre-defined statistics for specified dataset like /train_config/statistics. These are corresponding to the default values in <u>./se3set/model/layers/config.py</u>. You can determine them according to your datasets or simply use the default values. After this, you need to prepare a torch_geometric dataset for training.

For QM9, MD17, MD22 datasets experiments in our paper, we have provided a python script to preprocess these datasets. You can run 

```bash
python fragment_dataset.py -c [training-config-filepath]
```

to get the dataset saved at `data_params` `root`. Meanwhile, it will generate the statistics saved at <u>./train_config/statistics/<`dataset_name`>.yaml</u>.

### Train

To train an SE3Set model for your dataset, simply run

```bash
python train.py train.py --conf <training-config-filepath> --conf_dataset <statistics-filepath> 
```

For example, to train the MD17 Aspirin dataset, run

```bash
python train.py --conf ./train_config/md17/aspirin_template.yaml --conf_dataset ./train_config/statistics/MD17_Aspirin.yaml
```

If you want to load a checkpoint to continue training, you can run

```bash
python train.py train.py --conf <training-config-filepath> --conf_dataset <statistics-filepath> --load_ckpt_path <load-checkpoin-path> --suffix_continue_path <training-object-suffix-name>
```

## 📖   References

If you wish to cite our work, please do so as follows

```tex
@article{wu2024se3set,
  title={SE3Set: Harnessing equivariant hypergraph neural networks for molecular representation learning},
  author={Wu, Hongfei and Wu, Lijun and Liu, Guoqing and Liu, Zhirong and Shao, Bin and Wang, Zun},
  journal={arXiv preprint arXiv:2405.16511},
  year={2024}
}
```

