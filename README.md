# FashionNLP
FashionBrain D2.1: Named Entity Recognition and Linking Methods
EU project 732328: "Fashion Brain"

In this repository, we provide a natural language processing tool called FashionNLP which is specially designed for fashion textual data. This tool extends existing state of the art NER technique to fashion application. More specifically, FashionNLP has three main components: NER, where fashion entities are recognized on textual data, NEL, where we link the fashion entity to the FashionBrain taxonomy and finally, in case the fashion entity does not exist in the FashionBrain taxonomy, we add it to the taxonomy.

# Getting Started
This project requires PyTorch 0.4+ and Python 3.6+. you need to install Flair using this command
```
pip install flair
```
Then, you can start using the FashionNLP package:
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

- The arguments needed to train an LSTM-CRF model are:
	 - data folder: Path to input folder containing training and testing sets
	 - embedding: Type of embedding and it could be one of the three options:
	‘no char’, ‘char’ or ‘flair’
	The word ‘fashion’ should be filled with either white or black
	RGB # F1
	 - epochs: Number of epochs to train the chosen model
	 - output folder: Path to the output folder to save three files: loss.tsv contains the accuracy measures in each epoch, test.tsv contains the the testing set with model labels and training.log contains the log history.

 # Example:
   ``` bash 
      python lstm_fashion.py --data_folder '../data/lstm_input' --embedding 'no_char' --epochs 150 --output_folder '../output'
   ```
  - The arguments needed to use the bootstrap approach are:
	– model: Path to folder containing the model to load
	– first iteration: Path to folder containing the first iteration input data
	– second iteration: Path to folder containing the second iteration input data
	– epochs: Number of epochs to train the chosen model
	– retrained: Path to the data file trained in the second iteration
	– output folder: Path to the output folder to save three files

 # Example:
   ``` bash 
      python bootsrap_lstm.py --model '../output/no_char_1st_iter/final-model.pt' --first_iteration '../data/lstm_input' --second_iteration '../data/lstm_bootstrap' --epochs 100 --retrained '../data/lstm_bootstrap/retrained_data.tsv' --output_folder '../output/no_char_2nd_iter'
   ```
  - The arguments needed to enrich the FashionBrain taxonomy:
  	– taxonomy: Path to the FashionBrain taxonomy
	– test result: Path to the file containing the testing result filled with orange. Web safe RGB # F16823

   ``` bash 
      python taxonomy_matching.py --taxonomy '../data/enrichment_input/FBtaxonomy.csv' --test_result '../data/enrichment_input/test_result.txt'
   ```
