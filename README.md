# Query-and-reArrange
[![arXiv](https://img.shields.io/badge/arXiv-2306.01635-brightgreen.svg?logo=arXiv&style=flat-round)](https://arxiv.org/abs/2306.01635)
[![GitHub](https://img.shields.io/badge/GitHub-demo%20page-blue?logo=Github&style=flat-round)](https://zhaojw1998.github.io/Query_and_reArrange)
[![Colab](https://img.shields.io/badge/Colab-tutorial-blue?logo=googlecolab&style=flat-round)](https://colab.research.google.com/drive/1N3XeEfTCWNLTuBp9NWPwzW-hq7Ho7nQA?usp=sharing)

Repository for Paper: Zhao et al., [Q&A: Query-Based Representation Learning for Multi-Track Symbolic Music re-Arrangement](https://arxiv.org/abs/2306.01635), in IJCAI 2023 Special Track for AI the Arts and Creativity.

### Demo page
[https://zhaojw1998.github.io/Query_and_reArrange](https://zhaojw1998.github.io/Query_and_reArrange)

### Code and File Directory
This repository is organized as follows:
```
root
  ├──checkpoints/         model checkpoints
  │    
  ├──data/                processed data and pre-processing scripts
  │    
  ├──demo/                demo save directory
  │       
  ├──dl_modules/          Q&A model's sub-modules
  │    
  ├──utils/               scripts for utility functions
  │    
  ├──dataset.py           dataset and loader
  │   
  ├──model.py             Q&A model
  │   
  ├──train.py             traning script
  │ 
  └──inference.ipynb      tutorial for running the model
```

## How to run
* Q&A is now on [Google Colab](https://colab.research.google.com/drive/1N3XeEfTCWNLTuBp9NWPwzW-hq7Ho7nQA?usp=sharing), where you can quickly test our model online.
* Alternatively, follow the guidance in [`./inference.ipynb`](./inference.ipynb) offline for more in-depth testing. 
* If you wish to train our model from scratch, run [`./train.py`](./train.py). You may wish to configure a few params such as `BATCH_SIZE` from the beginning of the script. When `DEBUG_MODE`=1, it will load a small portion of data and quickly run through for debugging purpose.
* Dependencies of our work includes [pytorch](https://pytorch.org/) (ver. >= 1.10), [pretty_midi](https://pypi.org/project/pretty_midi/), [scipy](https://pypi.org/project/scipy/), [tensorboard](https://pypi.org/project/tensorboard/), and [tqdm](https://pypi.org/project/tqdm/2.2.3/).

## Data
* For details about the data we use, please refere to [./data](./data).

## Contact
Jingwei Zhao (PhD student in Data Science at NUS)

jzhao@u.nus.edu

June. 04, 2022
 
