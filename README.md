# AM/α-AM

This repo covers the implementation of the following paper: 

**"Adaptive manifold for imbalanced transductive few-shot learning (WACV 2024)"** [Pre-print](https://arxiv.org/abs/2304.14281), [Paper](https://openaccess.thecvf.com/content/WACV2024/html/Lazarou_Adaptive_Manifold_for_Imbalanced_Transductive_Few-Shot_Learning_WACV_2024_paper.html)
<p align='center'>
  <img src='idea_am.png' width="800px">
</p>

## Abstract

Transductive few-shot learning algorithms have showed substantially superior performance over their inductive counterparts by leveraging the unlabeled queries at inference. However, the vast majority of transductive methods are evaluated on perfectly class-balanced benchmarks. It has been shown that they undergo remarkable drop in performance under a more realistic, imbalanced setting.

To this end, we propose a novel algorithm to address imbalanced transductive few-shot learning, named Adaptive Manifold. Our algorithm exploits the underlying manifold of the labeled examples and unlabeled queries by using manifold similarity to predict the class probability distribution of every query. It is parameterized by one centroid per class and a set of manifold parameters that determine the manifold. All parameters are optimized by minimizing a loss function that can be tuned towards class-balanced or imbalanced distributions. The manifold similarity shows substantial improvement over Euclidean distance, especially in the 1-shot setting.

Our algorithm outperforms all other state of the art methods in three benchmark datasets, namely miniImageNet, tieredImageNet and CUB, and two different backbones, namely ResNet-18 and WideResNet-28-10. In certain cases, our algorithm outperforms the previous state of the art by as much as $4.2\%$.

If you find this repo useful for your research, please consider citing the paper
```
@inproceedings{AMWACV2024,
  title={Iterative label cleaning for transductive and semi-supervised few-shot learning},
  author={Lazarou, Michalis and Stathaki, Tania and Avrithis, Yannis},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2024}
}
```

## 1. Getting started

**This code is based on the codebase of the NeuRIPS 2022 paper "Realistic evaluation of transductive few-shot learning". The following instructions are the same as the aforementioned paper.**

### 1.1 Quick installation (recommended) (Download datasets and models)
Download datasets and pre-trained models (checkpoints) from this [link][https://eur02.safelinks.protection.outlook.com/?url=https%3A%2F%2F1drv.ms%2Ff%2Fs!Ak0Hi3lyg2AOgsAVL_O-N7q1w20oOw%3Fe%3D99Stet&data=05%7C02%7Cm.lazarou%40surrey.ac.uk%7Cc434846dc78649fe886a08dc22547209%7C6b902693107440aa9e21d89446a2ebb5%7C0%7C0%7C638422993281252024%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C0%7C%7C%7C&sdata=itF%2FyGMQmSz6OvG1zVYQKa%2F67mROw%2BSeSaTNVtPgujE%3D&reserved=0]

#### 1.1.1 Place datasets
Make sure to place the downloaded datasets (data/ folder) at the root of the directory.

#### 1.1.2 Place models
Make sure to place the downloaded pre-trained models (checkpoints/ folder) at the root of the directory.

### 1.2 Manual installation
Follow instruction 1.2 of NeurIPS 2020 paper "TIM: Transductive Information Maximization" public implementation (https://github.com/mboudiaf/TIM) if facing issues with previous steps. Make sure to place data/ and checkpoints/ folders at the root of the directory.

### 2. Requirements
To install requirements:
```bash
conda create --name <env> --file requirements.txt
```
Where \<env> is the name of your environment

## 3. Reproducing the main results

Before anything, activate the environment:
```python
source activate <env>
```

To reproduce the imbalanced results from Tables 2. and 3. in the paper, from the src/ directory execute this python command. Use the --plc for plc pre-processing

* Please make sure to set up the correct path wherever is required in the config/<balanced/dirichlet>/base_config/<backbone>/<dataset> in every `base_config.yaml` file. 

```python
python main.py --backbone <resnet18/wideres> --config_path <path/to/config> --dataset <mini/cub/tiered> --method <method_name> --balancing dirichlet --phi mus+G+Wb --plc
```

To reproduce the balanced results from Tables 4. and 5. in the paper, from the src/ directory execute this python command. Use the --plc for plc pre-processing
```pythondirectory/
python main.py --backbone <resnet18/wideres> --config_path <path/to/config> --dataset <mini/cub/tiered> --method <method_name> --balancing balanced --phi mus+G+Wb --plc
```

## Contacts
For any questions, please contact:

Michalis Lazarou (ml6414@ic.ac.uk)  
Yannis Avrithis (yannis@avrithis.net)  
Tania Stathaki (t.stathaki@imperial.ac.uk)  


## Acknowlegements
[α-ΤΙΜ](https://github.com/oveilleux/Realistic_Transductive_Few_Shot)

[ΤΙΜ](https://github.com/mboudiaf/TIM)

[iLPC](https://github.com/MichalisLazarou/iLPC)

[S2M2_fewshot](https://github.com/nupurkmr9/S2M2_fewshot)




