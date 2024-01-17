# CrossT5
 
This is the basic implementation of our project: **CrossT5: complete mapped API parameters by code generation**.
- [CrossT5](#CrossT5)
  * [Description](#description)
  * [Project Structure](#project-structure)
  * [Datasets](#datasets)
  * [Reproducibility:](#reproducibility-)
    + [Environment:](#environment-)
    + [Preparation](#preparation)

## Description

`CrossT5` is  an automated technology that utilizes pre- and post-mapped API parameter documentation along with pre-mapped parameter sequences for parameter completion.

Specifically, CrossT5 consists of two components: cross-attention mechanism and feature fusion. It is capable of capturing information gaps between traditional methods and trained language models in this context, thereby achieving improved predictive performance.

## Project Structure

```
├─approaches  # MATL main entrance.
|	├─dataset  # contains the training dataset.
|	├─vocab  # contains the extracted vocabulary.
|	├─models.py  # the MATL fine tune model structure.
|	├─main.py  # the MATL entrance.
|	├─config      # Configuration for MATL fine tuning
|	├─preTrain_model      # pretrained models
|	├─train.py      # the train file
|	├─eval.py      # the test file
|	├─utils.py      # the util
├─pretrain      # pretrain code
├─Dataset    # the dataset on DL Framework and Java2swift.
|	├─DL2DL      # Tensorflow, Torch, CNTK && MXNet
|		├─dict      # the folder containing tf, torch, MXnet and CNTK API docunments and signatures
|		├─mappings.xlsx      # the mapping relationship from one Framework to Another
|		├─"frameworkname"_sig.txt      # the pre-processed signatures of framework APIs
|		├─"frameworkname"_desc.txt      # the pre-processed document of framework APIs
|		├─"frameworkname"_name.txt      # the pre-processed name of framework APIs
|	├─java2swift      # dataset of java2swift
|		├─mappings.xlsx      # the mapping relationship of java to swift
|		├─'*.txt'      # the pre-processed results of corresponding APIs
├─java jdt parser    # the parser to obtain java source code.
|	├─ParseJavaFile.java      # the tool
|	├─java_files_path.txt      #the java source code path
├─baselines      # the comparsion approaches
```

## Datasets

In total, we collected Compara which contains 736 data cases. All the data are uploaded into the "Datasets" directory. More detail can be found in our paper.

## Reproducibility

The pretrained models will be also uploaded soon.

### Environment

**Note:** 
- We attach great importance to the reproducibility of `CrossT5`. Here we list some of the key packages to reproduce our results. Also please refer to the `requirements.txt` file for package installation.

**Key Packages:**

Package            Version

h5py               3.7.0

ipdb               0.13.9

matplotlib         3.6.0

nltk               3.7

numpy              1.23.1

regex              2022.7.25

tables             3.7.0

torch              1.12.0+cu116

torchaudio         0.12.0+cu116

torchvision        0.13.0+cu116

tqdm               4.64.0

wordninja          2.0.0

transformer        4.37.0.dev0





