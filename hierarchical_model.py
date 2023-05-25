from typing import Dict
import torch
from transformers import AutoModel
from class_model import ClassModel
from utils import myutils

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class HierarchicalModel(torch.nn.Module):
    """
    HierarchicalModel is a class used for type 1 and type 2 models
    """

    def __init__(self, label_dict: Dict, mlm: str, prob_dropout=0.0, isManyMLM=False):
        super().__init__()
        self.label_dict = label_dict
        # If isManyMLM is true, then the model is of type 2 otherwise it is type 1
        self.isManyMLM = isManyMLM
        # The transformer model to use
        self.mlm = AutoModel.from_pretrained(mlm)

        # Find the size of the masked language model
        if hasattr(self.mlm.config, 'hidden_size'):
            self.mlm_out_size = self.mlm.config.hidden_size
        elif hasattr(self.mlm.config, 'dim'):
            self.mlm_out_size = self.mlm.config.dim
        else:  # if not found, guess
            self.mlm_out_size = 768

        # Find the max position embedding size of the mlm. That is, finding the max_length the data
        # can be for the given mlm
        if hasattr(self.mlm.config, 'max_position_embeddings'):
            # We subtract -2 to account for cls etc.
            self.max_length = self.mlm.config.max_position_embeddings - 2
        else:
            # if not found, guess
            self.max_length = 512

        self.dropout = torch.nn.Dropout(prob_dropout)
        self.taskA = torch.nn.Linear(self.mlm_out_size, label_dict['a'])

        # We check if there is more than one set of labels - that is if the model is actually hierarchical
        if len(label_dict) > 1:
            # If model is of type 2 (isManyMLM=True) then we create a new ClassModel, otherwise we
            # create a new linear layer
            self.taskB = ClassModel(label_dict['b'] - 1, mlm, 0.0) if isManyMLM else \
                torch.nn.Linear(self.mlm_out_size, label_dict['b'] - 1)

        # We check if there is more than two sets of labels, in order to see if task c is necessary.
        if len(label_dict) > 2:
            if isManyMLM:
                # If model is of type 2 (isManyMLM=True) then we create new ClassModels, otherwise we
                # create new linear layers
                self.taskC = torch.nn.ModuleDict({
                    '1': ClassModel(label_dict['c'][1], mlm),
                    '2': ClassModel(label_dict['c'][2], mlm),
                    '3': ClassModel(label_dict['c'][3], mlm),
                    '4': ClassModel(label_dict['c'][4], mlm),
                })
            else:
                self.taskC = torch.nn.ModuleDict({
                    '1': torch.nn.Linear(self.mlm_out_size, label_dict['c'][1]),
                    '2': torch.nn.Linear(self.mlm_out_size, label_dict['c'][2]),
                    '3': torch.nn.Linear(self.mlm_out_size, label_dict['c'][3]),
                    '4': torch.nn.Linear(self.mlm_out_size, label_dict['c'][4]),
                    # '0': torch.nn.Linear(self.mlm_out_size, label_dict['c'][0]),
                })
            # out_dim tell us the final dimension of model
            self.out_dim = sum(list(self.label_dict['c'].values()))

    def forward(self, inp: torch.tensor):
        # We ensure that the inp does not exceed the max length
        inp = inp[:, : self.max_length]
        # We get the size of the batch from the inp
        batch_size = inp.size()[0]

        # Run transformer model on input
        mlm_out = self.mlm(inp)
        # Keep only the last layer: shape=(batch_size, max_len, DIM_EMBEDDING)
        mlm_out = mlm_out.last_hidden_state
        # Keep only the output for the first ([CLS]) token: shape=(batch_size, DIM_EMBEDDING)
        mlm_out = mlm_out[:, :1, :].squeeze()

        out_a = self.taskA(mlm_out)

        out_a = self.dropout(out_a)
        # out_a has shape: (batch_size, label_dict['a']). We are interested in finding most
        # probable label, so we are looking in the in dimension 1.
        task_a_pred = torch.argmax(out_a, 1)
        # We initialize the tensors for task b and c to be a zero tensor.
        if len(self.label_dict) > 1:
            out_b = torch.zeros((batch_size, self.label_dict['b']), device=DEVICE)
        if len(self.label_dict) > 2:
            out_c = torch.zeros((batch_size, sum(self.label_dict['c'].values())), device=DEVICE)

        # We iterate through the entire batch now
        for i in range(batch_size):
            # If task a predicted non-sexist, then we will not add any values to the tensors for
            # task b or c and simply continue to the next index in the batch.
            if task_a_pred[i] == 0:
                continue
            elif len(self.label_dict) > 1:
                # We check if the model is of type 1 or 2
                if self.isManyMLM:
                    # If the model is of type 2, we run ClassModel on a single datapoint.
                    # ClassModel generally needs to be of shape: (batch_size, max_len).
                    # In order to ensure that the data is of this shape, we reshape it
                    # such that it goes from shape: (max_len,) to (1,max_len)
                    task_b_out = self.taskB(torch.reshape(inp[i], (1, inp[i].size(dim=0))))
                else:
                    task_b_out = self.taskB(mlm_out[i])
                for b in range(len(task_b_out)):
                    # Since our linear layer is of length 4, but our out tensor is of length 5,
                    # we add one to the idx of the out_b, we want out_b[0] to be 0, since we
                    # do not want our models to be able to predict task b to be non-sexist
                    # if task a said it was sexist
                    out_b[i][1 + b] = task_b_out[b]
                if len(self.label_dict) > 2:
                    # We add one to the task_b_pred since our task_b_out is of length 4.
                    # This means that if label 1 was the best, torch_argmax would return 0
                    # if we didn't add +1 to this prediction
                    task_b_pred = 1 + torch.argmax(task_b_out, 0).item()
                    # If the model is of type 2, we run ClassModel on a single datapoint.
                    # ClassModel generally needs to be of shape: (batch_size, max_len).
                    # In order to ensure that the data is of this shape, we reshape it
                    # such that it goes from shape: (max_len,) to (1,max_len)
                    task_c_out = self.taskC[str(task_b_pred)](
                        torch.reshape(inp[i], (1, inp[i].size(dim=0)))) if self.isManyMLM \
                        else self.taskC[str(task_b_pred)](mlm_out[i])
                    # offset_idx tells us where in the task_c tensor the task_c_out values should be placed
                    offset_idx = myutils.get_c_offset(self.label_dict['c'], task_b_pred)
                    for c in range(len(task_c_out)):
                        out_c[i][offset_idx + c] = task_c_out[c]
        if len(self.label_dict) > 2:
            return out_a, out_b, out_c
        elif len(self.label_dict) > 1:
            return out_a, out_b
        return out_a
