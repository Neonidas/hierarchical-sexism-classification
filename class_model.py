import torch
from transformers import AutoModel


class ClassModel(torch.nn.Module):
    """
    ClassModel is a class used for type 0 models (and within the use of type 2 models)
    """
    def __init__(self, nlabels: int, mlm: str, prob_dropout=0.0):
        super().__init__()
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
            self.max_length = self.mlm.config.max_position_embeddings-2
        else:
            # if not found, guess
            self.max_length = 512

        self.dropout = torch.nn.Dropout(prob_dropout)
        self.hidden_to_label = torch.nn.Linear(self.mlm_out_size, nlabels)

    def forward(self, inp: torch.tensor):
        # We ensure that the inp does not exceed the max length
        inp = inp[:, :self.max_length]

        # Run transformer model on input
        mlm_out = self.mlm(inp)

        # Keep only the last layer: shape=(batch_size, max_len, DIM_EMBEDDING)
        mlm_out = mlm_out.last_hidden_state
        # Keep only the output for the first ([CLS]) token: shape=(batch_size, DIM_EMBEDDING)
        mlm_out = mlm_out[:, :1, :].squeeze()

        output_scores = self.hidden_to_label(mlm_out)

        output_scores = self.dropout(output_scores)

        return output_scores
