This is the repository of our work appeared in ICML 2024, `MaSS: Multi-attribute Selective Suppression for Utility-preserving Data Transformation from an Information-theoretic Perspective`.

[Paper](https://arxiv.org/abs/2405.14981).

If you use the code from this repo, please cite our work. Thanks!

```
@inproceedings{
chen2024mass,
title={Ma{SS}: Multi-attribute Selective Suppression for Utility-preserving Data Transformation from an Information-theoretic Perspective},
author={Yizhuo Chen and Chun-Fu Chen and Hsiang Hsu and Shaohan Hu and Marco Pistoia and Tarek F. Abdelzaher},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=61RlaY9EIn}
}
```

# Setup Env and Installation


```bash
conda create -y -n mass python=3.6
conda activate mass
pip install -r requirements.txt
```

# Data Preprocessing

## AudioMNIST

Download the dataset from the source website:

```
git clone https://github.com/soerenab/AudioMNIST.git
```

and then, run `tools/audiomnist_preprocess.py` for preprocess.

```
python3 tools/audiomnist_preprocess.py --audio_path AudioMNIST/
```

The created features are stored in `./data/audiomnist/audiomnist_train_fp16.pkl` and `./data/audiomnist/audiomnist_val_fp16.pkl`


## MotionSense

Download the dataset from the source website:

```
git clone https://github.com/mmalekzadeh/motion-sense.git
```

and then, unzip the file inside the repo `data/A_DeviceMotion_data.zip`, and run the `tools/motionsense_preprocess.py` to preprocess data.
```
cd ./motion-sense/data/
unzip A_DeviceMotion_data.zip 
cd ../../
python3 tools/motionsense_preprocess.py --data_path ./motion-sense/ --output_path ./data/motionsense/
```

# Run Our Algorithm, MaSS

The process to run MaSS is similar to each dataset, it involves a couple of steps:
1. Train the attribute classifiers for each attribute (U and S in the paper)
2. Train an unannotated attribute model with InfoNCE (F in the paper)
3. Train the MaSS with all above pretrained models 
4. Train another set of attribute classifiers with different settings for the utilities, e.g. random seed, for the use of finetuning the utility models over transformed data (we embed this step into MaSS training).


We have the training scripts for above processing in the `scripts` folder for both `audiomnist` and `motionsense` datasets with proper comments. The scripts contain training commands for the settings of Table 2-4 in the paper.

For example, to run AudioMNIST experiments:
```
bash scripts/audiomnist.sh
```
and the results are stored in `icml_results` folder.


## m, n setting of MaSS
As described in the paper, the users can control the mutual information over sensitive and utility via `loss-m` and `loss-n` setting, if your setting violate the constraints, 
MaSS will stop the program if the settings violate the constraints.


## Results
For the accuracy of each attribute classifier, you can get the performance at the end of `log.txt` file in the `$OUTPUT_DIR` folder; while for MaSS, the performance of sensitive attribute(s) and preserved attribute(s) are recorded in `mass_log.txt` in the output folder.


# Contact
If you have any questions, please feel free to contact us through email (richard.cf.chen@jpmchase.com).

