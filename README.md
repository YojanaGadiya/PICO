# PICO

The repository contains the code to train and evaluate an entity tagging system for the medical domain. 

## Contents:

* spans: the code to train and evaluate the system that tags entities as _participants_, _interventions_, _outcomes_, or _none_

  * data: contains the data to train and evaluate the model (train and dev datasets are agrregated crowdsourced data; test data are expert labeled data)
  
  * model: support methods to run the system
  
  * results/test: trained model
  
  * build_data.py: creates the datafiles necessary to train the model from _ebm_nlp_1_00_ (needs to be extracted before running)
  
  * train.py: trains the model
  
  * evaluates the model
 
* hier_labels: the code to train and evaluate three hierarchical label tagger for _participants_, _interventions_, _outcomes_

The contents are the same as in _spans_, but each folder contains three additional directories (_participants_, _interventions_, _outcomes_) to store the files specific for each of the three systems.


## Running the systems

### To run from scratch 

```
make glove 
```
```
make run
```

### To run with the pre-built data in ../data/

```
python train.py <...|participants|interventions|outcomes> 
```
*use the arguments only for hieratchical labels*

### To run the docker

From the PICO directory, to build:

```
sudo docker build -t <docker_name> .
```
to run: 
```
sudo docker run <docker_name>
```
