# Query-and-reArrange
<a href="https://colab.research.google.com/drive/1N3XeEfTCWNLTuBp9NWPwzW-hq7Ho7nQA?usp=sharing" rel="nofollow"><img src="https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" style="max-width: 100%;"></a>

Repository for Paper: Zhao et al., [Q&A: Query-Based Representation Learning for Multi-Track Symbolic Music re-Arrangement](https://arxiv.org/abs/2306.01635), in IJCAI 2023 Special Track for AI the Arts and Creativity.

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

## TBD
* Demo page and more detailed instructions will come out soon!

## Contact
Jingwei Zhao (PhD student in Data Science at NUS)

jzhao@u.nus.edu

June. 04, 2022
 
