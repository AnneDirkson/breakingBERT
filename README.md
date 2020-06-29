# Breaking BERT
This repository contains the code for Breaking BERT: Adversarial Attack for Named Entity Recognition

# Prerequisites 

For installing the required packages: 
#using pip
pip install --upgrade pip
pip install -r ./env/requirements.txt

#or using conda
conda env create -f ./env/environment.yml

Additionally you will need to install Pattern:
https://github.com/clips/pattern

As this is problematic with Python 3, save the Pattern repository in the working directory before installing with pip. (see their github for details).

# General Usage 

only_ent_ids, no_ent_ids, only_wrong_ids, out_texts = AdversarialBERT().main(test, modelpath,savepath, savepath2, 'wnut_labels.txt', devdata, traindata, testdata, random_attack =False, make_first_prediction= True, entities = True)

* savepath = the savepath of the adversarial examples
* savepath = the savepath of other output 
* test = the adversarial sample  (an example of input format provided under /data/WNUT_AdversarialSample.tsv) 
* devdata, traindata and testdata = the dev, train and test data of the dataset which are used in order to generate the entity lists for replacing entities. (Examples of input format provided under /data/)
*The sample, devdata, traindata and testdata should be loaded before running the script so the files should be provided, not the paths

* labels.txt is a txt file with all NER labels in the data (an example for WNUT is provided under /data/wnut_labels.txt). The code loads the file automatically. 
* random_attack is a parameter for performing a random attack instead of one that uses the importance ranking function. Default is set to False
* make_first_prediction is a parameter which can be used if running multiple instances with the same model and data. The initial predictions of the BERT model on the test set are automatically saved under ./first_preds_list and ./first_probs and can be reused.
* entities is the parameter that determines the type of attack. Set to True for entity attack and False for entity context attack. Set to None to not run either attack but only analyse which predictions are made for the test set and which sentences do not contain any entities or only entities.

## using the Jupyter Notebook 
/src/AdversarialAttackNER.ipynb
*Examples of usage can also be found in .ipynb


## using the .py file 
from AdversarialAttackNER import AdversarialBERT


# Data used in paper 

The data used in the paper is publicly available. For general NER, we used: 
* CONLL 2003 English data, available at: https://www.clips.uantwerpen.be/conll2003/ner/
* WNUT 2017 Emerging Entities data, available at: https://noisy-text.github.io/2017/

For biomedical NER, we used: 
* NCBI Disease corpus, available at: https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/
* BC5CDR corpus, available at: https://biocreative.bioinformatics.udel.edu/resources/
