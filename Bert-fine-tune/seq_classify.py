from __future__ import absolute_import, division, print_function


import logging
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import (DataLoader,
                             RandomSampler,
                             SequentialSampler,
                             TensorDataset)

# local imports
import bert_models
import tasks
import tools
from tools import (
    REGRESSION,
    CLASSIFICATION
)

import utils

log = logging.getLogger(__file__)
log.setLevel(logging.INFO)


def loadPreTrainedModel(args):
    fdir = args.pretrain_dir
    config, fweight = tools.getModelConfig(fdir)

    emb_size = args.emb_size
    num_labels = args.num_labels

    model = bert_models.BertForSequenceClassification(config,emb_size,num_labels)
    tools.doLoadModel(model, fweight)
    return model


def loadSeqClassifyModel(fdir,emb_size,num_labels):
    log.info("begin to load BertForSeqClassification model from: %s" % (fdir))
    config, fweight = tools.getModelConfig(fdir)
    model = bert_models.BertForSequenceClassification(config,emb_size,num_labels)

    state_dict = torch.load(fweight)
    model.load_state_dict(state_dict, strict=True)
    tools.printModelSize(model)
    log.info("finished loading model from: %s" % (fdir))
    return model

def restoreLabel(label):
    # return (label * 2.5 + 2.5)
    return label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()



def convertPairFeatures(examples, label_list, max_seq_length,
                        tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s.
    exmaples: are tasks.InputExamples;
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        if ex_index < 5:
            log.info("*** Example ***")
            log.info("guid: %s" % (example.guid))
            log.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            log.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            log.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            log.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            log.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))

    return features


def getTensorData(features, output_model=CLASSIFICATION):
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    dtype = torch.long
    if output_model == REGRESSION:
        dtype = torch.float
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=dtype)
    log.info("todel: shape=%s, %s" % (all_label_ids.shape, all_label_ids.dtype))

    tensor_data = TensorDataset(input_ids, input_mask, segment_ids,
                                all_label_ids)
    return tensor_data, all_label_ids


def trainEpoch(model, optimizer, loss_fn, device, data_loader,
                index, train_steps, prefix):
    # loss_fn = torch.nn.CosineEmbeddingLoss().to(device)
    counter2 = tools.Counter(train_steps, "Epoch")
    counter3 = tools.Counter(train_steps, "batch")

    model.train()
    total_loss = 0.0
    total_num = 0
    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)
        #print('batch:',batch)
        input_ids, input_mask, segment_ids,label_id = batch
        #print("label_ids",label_id)
        #logits = model(input_ids,attention_mask=input_mask)
        logits=model(input_ids,input_mask,segment_ids)
        logits=logits.softmax(dim=1)
        #print('logits:', logits)
        loss = loss_fn(logits,label_id)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        vloss = loss.item()
        num = label_id.size(0)
        counter2.update(vloss, num)
        counter3.update(vloss, num)
        total_loss += vloss
        total_num += num

        index += 1
        if index % 50 == 0:
            log.info("%s %s" % (prefix, counter3))
            counter3.reset()

    log.info("%s %s" % (prefix, counter2))
    return total_loss, total_num, index


def getLossFunction(device):
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    return loss_fn


def getTrainData(args):
    # 1. get data
    processor, output_mode = tools.getDataProcessor(args)
    train_examples = processor.get_train_examples(args.data_dir)

    # 2. get lable num
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = tools.getTokenizer(args)
    train_features = convertPairFeatures(train_examples, label_list,
        args.max_seq_length, tokenizer, output_mode)

    tensor_data, _ = getTensorData(train_features, output_mode)
    train_sampler = RandomSampler(tensor_data)
    train_dataloader = DataLoader(tensor_data, sampler=train_sampler,
                batch_size=args.train_batch_size)

    # 5. other info
    # compute the total steps
    steps = tools.computeSteps(len(train_examples), args)
    log.info("******** Training Data Information ***********")
    log.info("data_dir: %s" % (args.data_dir))
    log.info("\tNum Examples = %d" % (len(train_examples)))
    log.info("\tBatch Size = %d" % (args.train_batch_size))
    log.info("\tLearning Rate = %s" % (args.learning_rate))
    log.info("\tTotal Steps = %d" % (steps))
    log.info("\tNum labels = %d" % (num_labels))

    return train_dataloader, num_labels, steps


def getEvalData(args):
    #1. get data
    processor, output_mode = tools.getDataProcessor(args)
    eval_examples = processor.get_dev_examples(args.data_dir)

    # 2. get label type num
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # 3. convert tokens to IDs
    tokenizer = tools.getTokenizer(args)
    eval_features = convertPairFeatures(
        eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)

    # 4. convert tokens to tensors
    tensor_data, all_label_ids = getTensorData(eval_features, output_mode)
    eval_sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(tensor_data, sampler=eval_sampler,
                batch_size=args.eval_batch_size)

    # 5. other info
    # compute the total steps
    steps = int(len(eval_examples) / args.eval_batch_size)
    log.info("******** Testing Data Information ***********")
    log.info("data_dir: %s" % (args.data_dir))
    log.info("\tNum Examples = %d" % (len(eval_examples)))
    log.info("\tBatch Size = %d" % (args.eval_batch_size))
    log.info("\tTotal Steps = %d" % (steps))
    log.info("\tNum labels = %d" % (num_labels))

    return dataloader, num_labels, all_label_ids


def trainIt(args):
    tools.setupRandomSeed(args)
    device = tools.getDevice(args)
    loss_fn = getLossFunction(device)

    dataloader, _, train_steps = getTrainData(args)
    model = loadPreTrainedModel(args)
    model.to(device)

    optimizer = tools.getOptimizer(args, model, train_steps + 1.0)
    log.info("Begin to train the model.")
    model.train()
    # model.eval() # to disable Dropout and BN
    epochs = int(args.num_train_epochs)
    counter1 = tools.Counter(train_steps, "Over-all")
    i = 0
    for e in range(epochs):
        prefix = "[%d/%d]" % (e, epochs)
        log.info("xxxxxxx begin epoch %s xxxxxxxxxx" % (prefix))
        vloss, num, i = trainEpoch(model, optimizer, loss_fn, device,
                                dataloader, i, train_steps, prefix)
        counter1.update(vloss, num)
        # if e < epochs - 1:
        #     tools.saveModel(args, model, "epoch%d" % (e))
#        if e > 2 and e < epochs - 1:
#            evalIt(args, model)
        log.info("xxxxxxx End epoch %s xxxxxxxxxx" % (prefix))
    log.info("Finished training: %s" % (counter1.info()))
    tools.saveModel(args, model)
    torch.save(model.state_dict(),'/home/cy/ccd/Bert-Fine-Tune/Prism-fine-tune/finetune/model-state-dict/code-mrpc-weight-freeze.pt')
    return model


def evalIt(args, model):
    is_classification = (tools.getModelType(args) == CLASSIFICATION)

    # 1. get model and data
    device = tools.getDevice(args)
    if model is None:
        model = loadSeqClassifyModel(args.output_dir, args.emb_size, args.num_labels)
    model.eval()
    model.to(device)
    dataloader, _, all_label_ids = getEvalData(args)

    loss_fn = getLossFunction(device)

    # 2. do prediction
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    pred_labels=[]
    total = int(all_label_ids.size(0) / args.eval_batch_size)

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        print('eval batch:',batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            #logits = model(input_ids, attention_mask=input_mask)
            logits = model(input_ids, input_mask, segment_ids)
            logits = logits.softmax(dim=1).argmax(dim=1)
            pred_labels.append(logits.detach().numpy())
            #true_labels.append(batch_labels.detach().numpy())
            #print('logits:', logits)
            print(logits.shape)
            loss_t = loss_fn(logits, label_ids)
            #print("loss_t:" + str(loss_t.item()))
        eval_loss += loss_t.item()
        #print('eval_loss:',eval_loss)
        nb_eval_steps += 1
        if nb_eval_steps % 50 == 0:
            log.info("progress: %d/%d" % (nb_eval_steps, total))

        #part = loss_t.detach().cpu().numpy()
        #part=logits.detach().cpu().numpy()
        #print('part:',part)
        #if preds is None:
            #preds = part
        #else:
            #preds = np.append(preds, part, axis=0)
            #preds_list.append(preds)
            #preds=torch.max(logits, 1)[1].cuda()
            #print("preds:" + str(preds))
     # 3. compute the metrics
    eval_loss = eval_loss / nb_eval_steps
    log.info("finish evaluating the model: eval_loss=%.5f" % (eval_loss))
    if is_classification:
        preds = np.argmax(preds[0], axis=0)
        #preds = torch.argmax(preds[0].detach().cpu(),dim=-1)
    else:
        preds = np.squeeze(preds)

    task_name = args.task_name.lower()

    #preds = restoreLabel(preds)
    #preds_list.append(preds.detach().cpu().numpy())
    labels = restoreLabel(all_label_ids.numpy())
    print("preds:"+str(pred_labels))
    print("labels:"+str(labels))
    log.info("pred.shape=%s, label.shape=%s" % (preds.shape, labels.shape, ))
    #log.info("pred=%s\nlabel=%s" % (preds[0:10], labels[0:10]))
    result = tasks.compute_metrics(task_name, preds, labels)
    result['eval_loss'] = eval_loss

    # 4. save the evaluation result
    tools.saveEvalResult(args, result)
    return


def main():
    args = tools.setupArgs()
    tools.checkArgs(args)

    model = None
    if args.do_train:
        log.info("begin to train model for task: %s" % (args.task_name))
        model = trainIt(args)

    if args.do_eval:
        log.info("begin to evaluate model for task: %s" % (args.task_name))
        evalIt(args, model)

    return


if __name__ == "__main__":
    utils.setupLog()
    main()
