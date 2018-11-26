# Running instructions for assignment 4

There are 6 experiments that can be run (the links are to the folders containing the relevant docker file):

## Experiments with labels not manually curated

* span labeling, k=10, % = 10 (best out of (relatively) small dataset models, dataset labeled using three-fold training (repo here ./size_and_informativeness_experiments/support_scripts_and_training/training_model_adjusted_for_three_folds):

   https://github.com/maxaalexeeva/PICO/tree/master/size_and_informativeness_experiments/10k_10%25_experiments/spans_10_10

* second iteration with the training data relabeled using this script ./size_and_informativeness_experiments/relabeling_conll.ipynb:

   https://github.com/maxaalexeeva/PICO/tree/master/size_and_informativeness_experiments/10k_10%25_experiments/spans_10_10_relabeled_data1
   
* training using original labels provided by Nye et al:
   https://github.com/maxaalexeeva/PICO/tree/master/size_and_informativeness_experiments/10k_10%25_experiments/spans_10_10_with_orig_labels


## Experiments with labels manually curated (200 sentences)
  
* training using 200 sentences with the labels obtained from multifold training (best results from using the small dataset):
  https://github.com/maxaalexeeva/PICO/tree/master/size_and_informativeness_experiments/200_sentence_experiments/spans_uncurated
  
* training done using 200 manually labeled examples:
  https://github.com/maxaalexeeva/PICO/tree/master/size_and_informativeness_experiments/200_sentence_experiments/spans_curated

* (no real need to run, results are close to 0) automatically relabeled using the relabeling script from ./size_and_informativeness_experiments/relabeling_conll.ipynb:
  https://github.com/maxaalexeeva/PICO/tree/master/size_and_informativeness_experiments/200_sentence_experiments/spans_automatically_relabeled

# PICO tagger

The repository contains the code to train and evaluate an entity tagging system for the medical domain. The code has been downloaded and modified from https://github.com/bepnye/EBM-NLP; the code is discussed in https://arxiv.org/pdf/1806.04185.pdf .

## Contents

* spans: the code to train and evaluate the system that tags tokens as _participants_, _interventions_, _outcomes_, or _none_

  * data: contains the data to train and evaluate the model (train and dev datasets are aggregated crowdsourced data; test data are expert labeled data)
  
  * model: support methods to run the system
  
  * results/test: trained model
  
  * build_data.py: creates the datafiles necessary to train the model from _ebm_nlp_1_00_ (needs to be extracted before running)
  
  * train.py: trains the model
  
  * evaluate.py: evaluates the model
 
* hier_labels: the code to train and evaluate three hierarchical label tagger for _participants_, _interventions_, _outcomes_

The contents are the same as in _spans_, but each folder contains three additional directories (_participants_, _interventions_, _outcomes_) to store the files specific for each of the three systems.

* the hierarchical labels (from https://github.com/bepnye/EBM-NLP)

      participants/
        0: No label
        1: Age
        2: Sex
        3: Sample size
        4: Condition

      interventions/
        0: No label
        1: Surgical
        2: Physical
        3: Pharmacological
        4: Educational
        5: Psychological
        6: Other
        7: Control

      outcomes/
        0: No label
        1: Physical
        2: Pain
        3: Mortality
        4: Adverse effects
        5: Mental
        6: Other

## Requirements

* python3.X

* tensorflow1.10.0


## Running the systems

### To run from scratch 

* extract ebm_nlp_1_00.tar.gz

```
make glove 
```
```
make run
```


### To run with the pre-built data in ../data/
```
make glove 
```
*downloads Glove Vectors; only needs to be run once*
```
python train(1).py <...|participants|interventions|outcomes> 
```
*use the arguments only for hieratchical labels; files ending in 1 are the options for running spans*

### To evaluate 

``` 
python evaluate(1).py
```

### To evaluate the pretrained models using the docker

From the PICO directory, to build:

```
sudo docker build -t <docker_name> .
```
to run: 
```
sudo docker run <docker_name>
```




