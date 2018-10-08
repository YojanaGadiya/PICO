# PICO tagger

The repository contains the code to train and evaluate an entity tagging system for the medical domain. The code has been downloaded and modified from https://github.com/bepnye/EBM-NLP; the code is discussed in https://arxiv.org/pdf/1806.04185.pdf .

## Contents

* spans: the code to train and evaluate the system that tags tokens as _participants_, _interventions_, _outcomes_, or _none_

  * data: contains the data to train and evaluate the model (train and dev datasets are agrregated crowdsourced data; test data are expert labeled data)
  
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




