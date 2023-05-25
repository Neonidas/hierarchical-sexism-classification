# Sexism-Classification-Thesis

This repository has two model options, that is a hierarchical version, which will run through multiple different labels 
and a simple version in which you have to specify which task you are concerned with.

## Running the code
In order to run this code, remember to pip install the requirements file, that is 
```
pip install -r requirements.txt
```


### Preprocessing
If you intend on running this code on new data it is important to ensure that the data is of type csv. If you have data 
of type tsv, you can convert the data using our function `utils/convertTSVtoCSV.py`. To use this function, please run
the following line

```
python .\utils\convertTSVtoCSV.py {TSV data path}
```


Additionally, it is important that the given labels are INTs, this can be done many ways, we used our jupyter notebook 
`utils/label_str_to_int.ipynb`.

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
To run the base setup, that is the setup in which you need to specify the task you are working on.
This can be done using the following syntax
```
python3 main.py {TRAIN_DATA_PATH} {DEV_DATA_PATH} {TEST_DATA_PATH} {MLM} {TASK} 
```
At the moment, the code only allows up to 3 tasks which have to be named a,b or c.

### Running the singleMLM setup (type 1)
To run the singleMLM setup you have to use the following syntax
```
python3 main.py [TRAIN_DATA_PATH] [DEV_DATA_PATH] [TEST_DATA_PATH] [MLM]
```

### Running the multiMLM setup (type 2)
To run the multiMLM setup you have to use the following syntax
```
python3 main.py [TRAIN_DATA_PATH] [DEV_DATA_PATH] [TEST_DATA_PATH] [MLM] manyMLM
```

