# PICO

The repository contains the code to train and evaluate an entity tagging system for the medical domain. 

## Contents:

* spans: the code to train and evaluate the system that tags entities as _patients_, _interventions_, _outcomes_, or _none_

  ** data: contains the data to train and evaluate the model (train and dev datasets are agrregated crowdsourced data; test data are expert labeled data)
  
  ** model: support methods to run the system
  
  ** results/test: trained model
