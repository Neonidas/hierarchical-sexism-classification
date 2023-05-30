# Sexism-Classification-Thesis
Credit: Behlül Özcelik, Namra Imtiaz Özcelik, Sara Holst Winther, Rob Van der Goot (Supervisor) 

This repository contains the code and files used in the thesis paper "Hierarchical Sexism Detection and Classification on Social Media Data"
## Data
The data is sourced from the SemEval 2023 Task 10 - "Explainable Detection of Online Sexism" (EDOS) and can be found here: 
https://github.com/rewire-online/edos/tree/main

## How to run
In order to run this code, remember to pip install the requirements file:
```
pip install -r requirements.txt
```


### Preprocessing
If you intend on running this code on new data it is important to ensure that the data is of type csv. If you have data 
of type tsv, you can convert the data using our function `utils/convertTSVtoCSV.py`. To use this function, run
the following command

```
python .\utils\convertTSVtoCSV.py {TSV data path}
```


Additionally, it is important that the given labels are integers. We provide a jupyter notebook 
`utils/label_str_to_int.ipynb` that changes string labels to integer labels.

### Running the SVM baseline setup
To execute the SVM setup, run the following command:

```
python3 SVM/SVM_baseline.py [TYPE] [RANGE] [TASK]
```

TYPE: specify 'word' for word-level and 'char' for character-level n-grams.
RANGE: specify the range in which the n-grams should be in.
Example: 1,2 for range 1 through 2. 
Example: 1,1 for only unigram.
TASK: specify which task to solve for. Can be 'a', 'b' or 'c'.


### Running the transformer baseline setup (type 0)
To run the baseline setup, it is necessary to specify which subtask to solve.
This can be done using the following syntax
```
python3 main.py [TRAIN_DATA_PATH] [DEV_DATA_PATH] [TEST_DATA_PATH] [MLM] [TASK] 
```
At the moment, the code only allows up to 3 tasks which have to be named a, b or c.
Example:
```
python3 main.py path_to_train_data path_to_dev_data path_to_test_data GroNLP/hateBERT c
```

### Running the singleMLM setup (type 1)
To run the singleMLM setup, use the following syntax:
```
python3 main.py [TRAIN_DATA_PATH] [DEV_DATA_PATH] [TEST_DATA_PATH] [MLM]
```
Example:
```
python3 main.py path_to_train_data path_to_dev_data path_to_test_data GroNLP/hateBERT
```

### Running the multiMLM setup (type 2)
To run the multiMLM setup, use the following syntax:
```
python3 main.py [TRAIN_DATA_PATH] [DEV_DATA_PATH] [TEST_DATA_PATH] [MLM] manyMLM
```
Example:
```
python3 main.py path_to_train_data path_to_dev_data path_to_test_data GroNLP/hateBERT manyMLM
```



## Contents
### Analysis/
In the Analysis folder, you will find Jupyter notebooks containing a data exploration of the EDOS dataset and qualitative and quantitative analysis of our model.

### Annotations/
In the Annotations folder, we included our manual annotations of 100 data samples from training and test split each, along with Fleiss Kappa scores and a exploratory analysis for each set of annotations.

### SVM/
You can find our SVM implementation in the SVM folder

### label_preds/
In the label_preds folder, you can find our baseline and hierarchical model predictions, including a reddit and gab split.

### utils/
Various utility code and helper functions belong in the utils folder.

### Implementations
You can find our implementation of a hierarchical model in the ```hierarchical_model.py``` file. It only supports up to three hierarchical levels.



