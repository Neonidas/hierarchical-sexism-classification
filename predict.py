import torch
import sys
from utils import myutils
from transformers import AutoTokenizer
import logging
import transformers
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, precision_recall_fscore_support

logger = logging.getLogger(__name__)


BATCH_SIZE = 16
PAD = "[PAD]"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def predict(model, dataloader: DataLoader):
    """Predicting on the given data and produce classification report. Returns the macro f1 score"""
    model.eval()
    y_pred = []
    y_target = []
    for text_batch, label_batch in tqdm(dataloader):
        if not task:
            # If the model is of type 1 or 2, then we look at the labels of the last task and
            # predict on that.
            label_batch = [label[-1] for label in label_batch]
            label_batch = torch.stack(label_batch)
            batch_pred = torch.argmax(model.forward(text_batch)[-1], 1)
        else:
            batch_pred = torch.argmax(model.forward(text_batch), 1)
        y_pred.append(batch_pred.cpu())
        y_target.append(label_batch.cpu())
    # Takes the tensors of size = batch_size of creates individual tensors for each prediction and
    # target
    y_pred = [element for sublist in y_pred for element in sublist]
    y_target = [element for sublist in y_target for element in sublist]
    logger.info(classification_report(y_true=y_target, y_pred=y_pred, zero_division=0.0))
    # Whilst it would technically be possible to figure out the macro f1 score based on the
    # classification report, in order to get the exact f1 score, we use a different call in order
    # to get the precision and recall, which we in turn use to calculate the f1 score.
    precision, recall, fbeta_score, support = precision_recall_fscore_support(y_target, y_pred,
                                                                              average='macro',
                                                                              zero_division=0.0)
    logger.info('precision: ' + str(precision))
    logger.info('recall: ' + str(recall))
    macro_f1 = (2 * precision * recall) / (precision + recall)
    logger.info("macro f1: " + str(macro_f1))
    logger.info('fbeta_score: ' + str(fbeta_score))
    return macro_f1, y_pred


if __name__ == '__main__':
    start_time = datetime.now()
    # Turns off transformer warnings
    transformers.logging.set_verbosity_error()
    task = None
    manyMLM = False
    # Checks if model is of type 0 or type 2. When it is type 2, we set manyMLM to true.
    # When it is type 0, we set which task we are at.
    if len(sys.argv) > 3:
        if sys.argv[3] == 'manyMLM':
            manyMLM = True
        else:
            task = sys.argv[3]

    # The MLM we use determines which tokenizer we are using
    model_path = sys.argv[2]

    if task is None:
        if not manyMLM:
            task_type = "type1"
        else:
            task_type = "type2"
    else:
        task_type = task


    # Model class must be defined somewhere
    model = torch.load(model_path)
    model.eval()

    MLM = model.mlm.name_or_path
    path = myutils.check_if_path_exists(task_type, MLM, start_time)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        level=logging.INFO, handlers=[logging.FileHandler(path + '/predict_out.txt'),
                                                      logging.StreamHandler(sys.stdout)])

    tokenizer = AutoTokenizer.from_pretrained(MLM)

    # PAD is used to make arrays of tokens the same size for batching purposes
    PAD = tokenizer.pad_token_id
    myutils.set_pad_id(PAD)

    logger.info('Initializing dataloaders...')
    dev_dataset = myutils.CustomTextDataset(sys.argv[1], tokenizer, task)

    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=BATCH_SIZE,
                                collate_fn=myutils.collate_fn, drop_last=True)

    f1, y_preds = predict(model, dev_dataloader)
    myutils.createPredictedFile(MLM, task_type, start_time, y_preds)
    
    elapsed = datetime.now() - start_time
    logger.info("Elapsed time: " + str(elapsed))
    logger.info("Run complete")
