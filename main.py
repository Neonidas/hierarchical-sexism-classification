import torch
import sys
from utils import myutils
from transformers import AutoTokenizer
import logging
import transformers
from datetime import datetime
from hierarchical_model import HierarchicalModel
from class_model import ClassModel
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, precision_recall_fscore_support

logger = logging.getLogger(__name__)

torch.manual_seed(8446)  # set seed for consistency

BATCH_SIZE = 16
LEARNING_RATE = 1e-5
EPOCHS = 20
DROPOUT_PROB = 0.4
PAD = "[PAD]"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# This code needs a list of input parameters,
# that being [TRAIN_DATA_PATH] [DEV_DATA_PATH] [TEST_DATA_PATH] [MLM]
if len(sys.argv) < 4:
    logger.info('Please provide path to training, development and test data as well as the mlm you want '
          'to use, this could be bert-base-cased or vinai/bertweet-base or GroNLP/hateBERT or '
          'similar ones.')


def training(train_dataloader, dev_dataloader, loss_function):
    """
    Training the model, given the train dataloader, the dev dataloader and a loss function which is
    either a singular loss_function or a list of loss functions depending on which setup type we
    are using. It returns the best_model.
    """

    best_f1 = 0.0
    best_epoch = 0
    best_model = model

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        logger.info('=====================')
        logger.info('starting epoch ' + str(epoch))
        # This call tells pytorch that the model is in training mode. This tells the model that
        # dropout should be active here as well as having batch normalization function as it should
        # in training.
        model.train(True)

        # Loop over batches
        loss = 0
        match = 0
        total = 0
        # Tqdm is simply a progress bar which can be added to for loops.
        for text_batch, label_batch in tqdm(train_dataloader):
            # We explicitly set the gradients to zero at the beginning of each batch in order to
            # avoid an accumulated gradient on subsequent backward passes. In other words this call
            # ensures that our gradient does not become a combination of old gradients.
            optimizer.zero_grad()

            output_scores = model.forward(text_batch)
            if not task:
                # We create a list for the batch losses, instead of simply using a float, because
                # we need the tensors that the loss function creates in order to call backward on
                # the losses.
                batch_losses = []
                # In the case that our model is of type 1 or 2, that is a hierarchical setup, there
                # will be a loss for each task. The following for loop will run through all the
                # tasks and call the needed loss function.
                for i, out in enumerate(output_scores):
                    targets = [label[i] for label in label_batch]
                    targets = torch.stack(targets)
                    if i > 0:
                        # At all tasks except the first binary one, we have to specify which
                        # indices to ignore
                        targets = myutils.set_ignore_indices(out, targets)
                    batch_loss = loss_function[i](out, targets)
                    # If we have a batch in which we have to ignore everything (that is a batch
                    # where everything is classed as non-sexist) the loss will become nan, which is
                    # not helpful to our calculations, so these losses are not added to our batch
                    # loss. (simple reason: nan + 10000 = nan)
                    if not np.isnan(batch_loss.item()):
                        batch_losses.append(batch_loss)
                # Since the loss function returns a tensor, we use .item() in order to get the
                # float loss value of the given tensor.
                loss += sum(batch_losses).item()
                # We sum together the loss tensors in order to have one tensor to call backward on
                sum(batch_losses).backward()
            else:
                # In the case that our model is of type 0, the loss can simply be added as such
                batch_loss = loss_function(output_scores, label_batch)
                loss += batch_loss.item()
                batch_loss.backward()
            # myutils.plot_grad_flow(model.named_parameters())
            optimizer.step()

            # Update the number of correct labels and total labels
            # If the model is of type 1 or 2, then we look at the last tasks results.
            pred_labels = torch.argmax(output_scores, 1) if task else torch.argmax(output_scores[-1], 1)
            target_labels = label_batch if task else label_batch[-1]
            for gold_label, pred_label in zip(target_labels, pred_labels):
                total += 1
                if gold_label == pred_label:
                    match += 1

        logger.info(" ")

        logger.info(f'Finished epoch {epoch}')
        logger.info('Training loss: {:.2f}'.format(loss))
        logger.info('Acc(train): {:.2f}'.format(100 * match / total))

        logger.info('Evaluating on dev set')
        macro_f1, _ = predict(dev_dataloader, model)

        # Update the best f1, best model and best epoch
        if best_f1 < macro_f1:
            logger.info(f'Best model so far with macro f1: {macro_f1} on epoch {epoch}')
            best_f1 = macro_f1
            logger.info("Saving model")
            myutils.savingModels(model, MLM, start_time, task_type)
            best_epoch = epoch
        logger.info(" ")
    logger.info(f'Best epoch: {best_epoch}')
    return best_model


def predict(dataloader: DataLoader, pred_model=None):
    """Predicting on the given data and produce classification report. Returns the macro f1 score"""
    if pred_model is None:
        logger.info("Loading best model...")
        pred_model = torch.load(path + 'model.pt')
    pred_model.eval()
    y_pred = []
    y_target = []
    for text_batch, label_batch in tqdm(dataloader):
        if not task:
            # If the model is of type 1 or 2, then we look at the labels of the last task and
            # predict on that.
            label_batch = [label[-1] for label in label_batch]
            label_batch = torch.stack(label_batch)
            batch_pred = torch.argmax(pred_model.forward(text_batch)[-1], 1)
        else:
            batch_pred = torch.argmax(pred_model.forward(text_batch), 1)
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
    if len(sys.argv) > 5:
        if sys.argv[5] == 'manyMLM':
            manyMLM = True
        else:
            task = sys.argv[5]

    # The MLM we use determines which tokenizer we are using
    MLM = sys.argv[4]

    if task is None:
        if not manyMLM:
            task_type = "type1"
        else:
            task_type = "type2"
    else:
        task_type = task

    path = myutils.check_if_path_exists(task_type, MLM, start_time)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        level=logging.INFO, handlers=[logging.FileHandler(path + '/out.txt'),
                                                      logging.StreamHandler(sys.stdout)])


    logger.info('Pre-trained transformer used: ' + MLM)
    tokenizer = AutoTokenizer.from_pretrained(MLM)
    # PAD is used to make arrays of tokens the same size for batching purposes
    PAD = tokenizer.pad_token_id
    myutils.set_pad_id(PAD)

    logger.info('Initializing dataloaders...')
    train_dataset = myutils.CustomTextDataset(sys.argv[1], tokenizer, task)
    dev_dataset = myutils.CustomTextDataset(sys.argv[2], tokenizer, task)
    test_dataset = myutils.CustomTextDataset(sys.argv[3], tokenizer, task)

    # We use dataloaders to create minibatches of size batch_size.
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  collate_fn=myutils.collate_fn, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=BATCH_SIZE,
                                collate_fn=myutils.collate_fn, drop_last=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=BATCH_SIZE,
                                 collate_fn=myutils.collate_fn, drop_last=True)

    # The setup of the model depends on the type, so the following if statements deal with picking
    # the correct type and by extension creating the correct amount of loss functions
    if not task:
        NLABELS = myutils.find_NLABELS(test_dataset.labels)
        model = HierarchicalModel(NLABELS, MLM, DROPOUT_PROB, manyMLM)
    else:
        NLABELS = len(np.unique(test_dataset.labels))
        model = ClassModel(NLABELS, MLM, DROPOUT_PROB)

    logger.info('Initializing model...')
    model.to(DEVICE)
    class_weights = myutils.get_class_weights(train_dataset.labels if task else train_dataset.labels[-1])

    if task:
        loss_function = torch.nn.CrossEntropyLoss(class_weights)
    else:

        class_weights_a = myutils.get_class_weights(train_dataset.labels[0])
        class_weights_b = myutils.get_class_weights(train_dataset.labels[1])

        loss_function = [torch.nn.CrossEntropyLoss(class_weights_a),
                         torch.nn.CrossEntropyLoss(class_weights_b)]
        if len(train_dataset.labels) > 2:
            loss_function.append(torch.nn.CrossEntropyLoss(class_weights))

    logger.info('Training...')
    training(train_dataloader, dev_dataloader, loss_function)

    logger.info('Testing model on Test set')
    f1, y_preds = predict(test_dataloader)


    if task is None:
        if not manyMLM:
            task = "type1"
        else:
            task = "type2"

    myutils.createPredictedFile(MLM, task, start_time, y_preds)
    max_memory = None
    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated() * 1e-09

    elapsed = datetime.now() - start_time
    logger.info("Elapsed time: " + str(elapsed))
    # logger.info("Saving model")
    # myutils.savingModels(best_model, MLM, start_time, task)
    myutils.createLogFile(MLM, task_type, LEARNING_RATE, EPOCHS, BATCH_SIZE, f1, start_time, max_memory, elapsed)
    logger.info("Run complete")
