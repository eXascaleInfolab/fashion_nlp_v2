# FashionNLP
FashionBrain D2.1: Named Entity Recognition and Linking Methods
EU project 732328: "Fashion Brain"

In this repository, we provide a natural language processing tool called FashionNLP which is specially designed for fashion textual data. This tool extends existing state of the art NER technique to fashion application. More specifically, FashionNLP has three main components: NER, where fashion entities are recognized on textual data, NEL, where we link the fashion entity to the FashionBrain taxonomy and finally, in case the fashion entity does not exist in the FashionBrain taxonomy, we add it to the taxonomy.

# Getting Started
```
git clone https://github.com/FashionBrainTeam/fashion_nlp_v2
cd ./fashion_nlp_v2/
```
# Description of the FashionNLP Package
The ``fashionnlp'' package contains three folders:
 - src contains three python scripts:
    - lstm_fashion.py: the implementation of the LSTM-CRF models
    - bootsrap_lstm.py: the implementation of the bootstraping approach
    - taxonomy_matching.py: the implementation of the taxonomy enrichment
 - data contains the training set (fashion_items_train.txt), the testing set (fashion_items_test.txt) and the FashionBrain taxonomy (FBtaxonomy.csv)
- output contains the results of the bootstrap approach.

### Running the code 
To train an LSTM-CRF model:
   ``` bash 
      python lstm_fashion.py 
   ```
  To use the boostap approach:
   ``` bash 
      python bootsrap_lstm.py 
   ```
 To enrich the FashionBrain taxonomy:
   ``` bash 
      python taxonomy_matching.py 
   ```
