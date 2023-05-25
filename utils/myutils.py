import math
import os

import emoji
import numpy as np
import pandas as pd
import torch
from typing import List

from sklearn.utils import compute_class_weight
from torch.utils.data import Dataset
from transformers import AutoTokenizer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def tok(data: List[str], tokenizer: AutoTokenizer):
    """
    tok is a helper function that runs tokenizer.encode() on the data
    """
    tok_data = []

    for sent in data:
        if tokenizer.name_or_path != 'bert-base-cased':
            # In the papers related to bertweet and hateBERT, they specify that user references
            # where specified in their pretraining as well as urls. These specifications are
            # different from our datasets, so we set them here before we encode the tokens.
            # Additionally, they mention that they demojize their emojis, which we also do here.
            sent = emoji.demojize(sent)
            sent = sent.replace('[USER]', '@USER')
            if tokenizer.name_or_path == 'vinai/bertweet-base' or tokenizer.name_or_path == 'vinai/bertweet-large':
                sent = sent.replace('[URL]', 'HTTPURL')
            elif tokenizer.name_or_path == 'GroNLP/hateBERT':
                sent = sent.replace('[URL]', 'URL')
        tok_data.append(tokenizer.encode(sent))
    return tok_data

def find_task_labels(labels, task: str):
    """
    find_task_labels takes a set of labels and returns only the labels relevant to the given task

    NOTE: task HAS to be either "a", "b" or "c"
    """
    if task == "a":
        label = labels[0]
    elif task == "b":
        label = labels[1]
    elif task == "c":
        label = labels[2]
    return label


def read_data(path: str, column_name: str, task=None):
    """
    read_data takes a dataset path, a column name and optionally a task and returns the data at
    the given column and the labels, if task is specified then the labels in a series stating
    (idx, label) for each datapoint. If task has not been specified then the returned labels is a
    list of all the different label types each of which has a series setup like the individual
    one set when task is specified.

    NOTE: task HAS to be either "a", "b" or "c"
    """
    df = pd.read_csv(path)
    text_idx = df.columns.get_loc(column_name)
    data = df[column_name]
    num_cols = len(df.columns)
    labels = []
    for i in range(text_idx + 1, num_cols):
        labels.append(df.iloc[:, i])
    if task:
        label = find_task_labels(labels, task)
        return data, label
    return data, labels


def find_NLABELS(labels):
    """
    find_NLABELS iterates through all label types and returns the number of unique labels in for
    every different label type
    """
    NLABELS = {'a': len(labels[0].unique())}
    # NOTE: task_b unique labels includes the label 0
    task_b = len(labels[1].unique())
    NLABELS['b'] = task_b
    # We create a dictionary for the labels in c
    if len(labels) > 2:
        NLABELS['c'] = {}
        task_c = {}
        # We iterate through all labels in b
        for m in range(0, task_b):
            task_b_unique_label_m = labels[1].unique()[m]
            task_c_all_unique_labels = labels[2].unique()
            # We iterate through all labels in c
            for n in range(len(task_c_all_unique_labels)):
                # We iterate through all the entire dataset
                for i in range(len(labels[0])):
                    # At each datapoint we see if the label of task b is equal to the label in the
                    # primary for loop and if the task c label is equal the label of the label of the
                    # secondary for loop
                    if labels[1][i] == task_b_unique_label_m and labels[2][i] == task_c_all_unique_labels[n]:
                        # We check if the task b label is already in the dictionary of task c
                        if task_b_unique_label_m in task_c:
                            # We check if the specific task c label is already added as a value at the
                            # given key, if not then we append this value to the values of the given
                            # key (the key is the label from task b)
                            if task_c_all_unique_labels[n] not in task_c[task_b_unique_label_m]:
                                task_c[task_b_unique_label_m].append(task_c_all_unique_labels[n])
                        else:
                            # If the task b label is not in the dictionary of task c, then we
                            # initialize the key of the given task b label and add the task c
                            # label to this key
                            task_c[task_b_unique_label_m] = [task_c_all_unique_labels[n]]
            # We insert the amount of unique task c labels into NLABELS for the given task b label.
            NLABELS['c'][task_b_unique_label_m] = len(task_c[task_b_unique_label_m])
    return NLABELS


def get_class_weights(labels):
    """
    Given a list of labels get_class_weights will return a tensor of the class weights of the
    given labels.
    """
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(labels),
        y=labels)
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    return class_weights


def check_if_path_exists(task, mlm, time):
    """
    Checks if the path to the file placement exists, and returns the final path
    """
    path = "models/"
    if not os.path.exists(path):
        os.mkdir(path)
    task_type = task
    if task != "type1" and task != "type2":
        task_type = "type0"

    path = path + task_type + '/'
    if not os.path.exists(path):
        os.mkdir(path)

    transformer_variant = mlm.split('/')[-1]  # 'vinai/bertweet-base' --> 'bertweet-base/'
    path = path + transformer_variant + '/'

    if not os.path.exists(path):
        os.mkdir(path)

    path = path + time.strftime("%d.%m-%H.%M.%S") + '_' + task + '/'

    if not os.path.exists(path):
        os.mkdir(path)
    return path


def savingModels(model, MLM, time, task):
    """Save model in models folder with given format: models/[MLM]/[DATE]-[TASK]-[TIME].pt"""
    path = check_if_path_exists(task, MLM, time)
    model_path = path + 'model.pt'
    torch.save(model, model_path)


def createLogFile(MLM, task, lr, epochs, batch_size, f1_score, time, max_memory, elapsed):
    path = check_if_path_exists(task, MLM, time)
    path = path + 'log.txt'
    with open(path, 'w') as f:
        f.write("Macro F1-Score: " + str(f1_score) + "\n")
        f.write("Learning rate: " + str(lr) + "\n")
        f.write("Number of epochs: " + str(epochs) + "\n")
        f.write("Batch size: " + str(batch_size) + "\n")
        f.write("GPU memory (None if CUDA not available) : " + str(max_memory) + "\n")
        f.write("Elapsed time: " + str(elapsed) + "\n")
        f.close()


def createPredictedFile(MLM, task, time, preds):
    path = check_if_path_exists(task, MLM, time)
    path = path + 'label_pred_test.csv'
    with open(path, 'w') as f:
        f.write("idx,prediction\n")
        for i in range(len(preds)):
            f.write(str(i) + "," + str(preds[i].item()) + "\n")
        f.close()


def collate_fn(batch):
    """
    collate_fn is used when having batches of varying size, in our case it is possible that the
    final batch is smaller that the previous batches and in order for this to function, we have
    to collate the batches. The collate_fn function takes a batch as input and returns tensors
    for the data and for the labels.
    """
    text_tensors, label_tensors = [], []
    for text, label in batch:
        text_tensors.append(torch.tensor(text, dtype=torch.long, device=DEVICE))
        label_tensors.append(torch.tensor(label, dtype=torch.long, device=DEVICE))
    # Pad_sequence ensures that each datapoint in the batch have the same length
    text_tensors = torch.nn.utils.rnn.pad_sequence(text_tensors, batch_first=True, padding_value=get_pad_id())
    label_tensors = torch.stack(label_tensors)
    return text_tensors, label_tensors


def set_ignore_indices(prev_out, labels):
    """
    set_ignore_indices takes the output of the previous task and sets the label to be -100 if the
    tensor is a zero-tensor and returns the edited labels.
    """
    for i, out_tensor in enumerate(prev_out):
        if torch.equal(out_tensor, torch.zeros_like(out_tensor)):
            labels[i] = -100
    return labels


pad = 0


def get_c_offset(labels, b_prediction):
    """
    get_c_offset calculates where in the c tensor the given predictions need to be placed.
    """
    offset = 0
    for category in range(b_prediction):
        offset += labels[category]
    return offset


def set_pad_id(PAD):
    """
    set_pad_id sets the global pad to be equal to the input PAD value.
    """
    global pad
    pad = PAD


def get_pad_id():
    """
    get_pad_id returns the global pad value.
    """
    global pad
    return pad


class CustomTextDataset(Dataset):
    """
    CustomTextDataset is a class which enables the codebase to function in type 0, as well as
    type 1 and 2.
    """

    def __init__(self, csv_file_path, tokenizer, task=None):
        """
        When initializing a CustomTextDataset, the data of the given path will be read and the data
        will be tokenized. If task is not specified the model will be of type 1 or 2.
        """
        self.texts, self.labels = read_data(csv_file_path, 'text', task)
        self.texts = tok(self.texts, tokenizer)
        self.task = task

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.task:
            label = self.labels[idx]
        else:
            if len(self.labels) > 2:
                label = [self.labels[0][idx], self.labels[1][idx], self.labels[2][idx]]
            elif len(self.labels) > 1:
                label = [self.labels[0][idx], self.labels[1][idx]]
        return text, label
