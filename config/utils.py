from typing import List
from common import Instance
import torch.optim as optim
import pickle
import os.path
from config import PAD, ContextEmb, Config
import torch
import torch.nn as nn


def log_sum_exp_pytorch(vec: torch.Tensor) -> torch.Tensor:
    """
    Calculate the log_sum_exp trick for the tensor.
    :param vec: [batchSize * from_label * to_label].
    :return: [batchSize * to_label]
    """
    maxScores, idx = torch.max(vec, 1)
    maxScores[maxScores == -float("Inf")] = 0
    maxScoresExpanded = maxScores.view(vec.shape[0] ,1 , vec.shape[2]).expand(vec.shape[0], vec.shape[1], vec.shape[2])
    return maxScores + torch.log(torch.sum(torch.exp(vec - maxScoresExpanded), 1))


def batching_list_instances(config: Config, insts: List[Instance], is_soft=False, is_naive=False):
    train_num = len(insts)
    batch_size = config.batch_size
    total_batch = train_num // batch_size + 1 if train_num % batch_size != 0 else train_num // batch_size
    batched_data = []
    for batch_id in range(total_batch):
        one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        batched_data.append(simple_batching(config, one_batch_insts, is_soft, is_naive))
    return batched_data


def simple_batching(config, insts: List[Instance], is_soft=False, is_naive=False):
    batch_size = len(insts)
    batch_data = insts
    label_size = config.label_size

    word_seq_len = torch.LongTensor(list(map(lambda inst: len(inst.input.words), batch_data)))
    max_seq_len = word_seq_len.max()
    char_seq_len = torch.LongTensor([list(map(len, inst.input.words)) + [1] * (int(max_seq_len) - len(inst.input.words)) for inst in batch_data])
    max_char_seq_len = char_seq_len.max()

    word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    label_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_char_seq_len), dtype=torch.long)

    annotation_mask = None
    if batch_data[0].is_prediction is not None:
        annotation_mask = torch.zeros((batch_size, max_seq_len, label_size), dtype=torch.long)

    trigger_label_seq_tensor = None
    if batch_data[0].trigger_label is not None and not is_naive:
        trigger_label_seq_tensor = torch.LongTensor(list(map(lambda inst: inst.trigger_label, batch_data)))

    trigger_vec_tensor = None
    if batch_data[0].trigger_vec is not None and not is_naive:
        trigger_vec_tensor = []

    context_emb_tensor = None
    if config.context_emb:
        emb_size = config.context_emb_size
        context_emb_tensor = torch.zeros((batch_size, max_seq_len, emb_size))

    for idx in range(batch_size):
        word_seq_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].word_ids)
        if batch_data[idx].output_ids:
            label_seq_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].output_ids)
        if config.context_emb:
            sentence = batch_data[idx].input.words
            emb = config.bert_model_matrix.get(' '.join(sentence))
            if emb is None:
                emb = config.bert_model.bert_embedding(sentence)
            context_emb = emb + (max_seq_len - len(sentence)) * [[0] * config.context_emb_size]
            context_emb_tensor[idx, :, :] = torch.FloatTensor(context_emb)
            # context_emb_tensor[idx, :word_seq_len[idx], :] = batch_data[idx].elmo_vec[1:batch_data[idx].elmo_vec.shape[0] - 1, :]
        if batch_data[idx].is_prediction is not None:
            for pos in range(len(batch_data[idx].input)):
                if batch_data[idx].is_prediction[pos]:
                    annotation_mask[idx, pos, :] = 1
                    annotation_mask[idx, pos, config.start_label_id] = 0
                    annotation_mask[idx, pos, config.stop_label_id] = 0
                else:
                    annotation_mask[idx, pos, batch_data[idx].output_ids[pos]] = 1
            annotation_mask[idx, word_seq_len[idx]:, :] = 1
        for word_idx in range(word_seq_len[idx]):
            char_seq_tensor[idx, word_idx, :char_seq_len[idx, word_idx]] = torch.LongTensor(batch_data[idx].char_ids[word_idx])
        for wordIdx in range(word_seq_len[idx], max_seq_len):
            char_seq_tensor[idx, wordIdx, 0: 1] = torch.LongTensor([config.char2idx[PAD]])
        if batch_data[idx].trigger_vec is not None and is_naive is False:
            trigger_vec_tensor.append(batch_data[idx].trigger_vec)

    word_seq_tensor = word_seq_tensor.to(config.device)
    label_seq_tensor = label_seq_tensor.to(config.device)
    char_seq_tensor = char_seq_tensor.to(config.device)
    word_seq_len = word_seq_len.to(config.device)
    char_seq_len = char_seq_len.to(config.device)
    annotation_mask = annotation_mask.to(config.device) if annotation_mask is not None else None

    trigger_label_seq_tensor = trigger_label_seq_tensor.to(config.device) if trigger_label_seq_tensor is not None else None
    trigger_position_seq_tensor = [batch_data[idx].trigger_positions for idx in range(batch_size)] if batch_data[0].trigger_positions is not None else None

    if is_soft:
        return word_seq_tensor, word_seq_len, context_emb_tensor, char_seq_tensor, char_seq_len, annotation_mask, label_seq_tensor, torch.stack(trigger_vec_tensor)
    return word_seq_tensor, word_seq_len, context_emb_tensor, char_seq_tensor, char_seq_len, annotation_mask, label_seq_tensor, trigger_position_seq_tensor, trigger_label_seq_tensor

def lr_decay(config, optimizer: optim.Optimizer, epoch: int) -> optim.Optimizer:
    """
    Method to decay the learning rate
    :param config: configuration
    :param optimizer: optimizer
    :param epoch: epoch number
    :return:
    """
    lr = config.learning_rate / (1 + config.lr_decay * (epoch - 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('learning rate is set to: ', lr)
    return optimizer


def load_bert_vec(file: str, insts: List[Instance]):

    if os.path.exists(file.split('.')[0]+'.vec'):
        f = open(file.split('.')[0]+'.vec', 'rb')
        bert_embedding = pickle.load(f)
        f.close()
        size = 0
        for vec, inst in zip(bert_embedding, insts):
            inst.elmo_vec = vec
            size = vec.shape[1]
            assert (vec.shape[0] == len(inst.input.words)+2)
        return size
    return 0


def get_optimizer(lr, weight_decay, config,  model: nn.Module, name=None):
    params = model.parameters()
    if name is not None and name == "adam":
        print("Using Adam")
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name is not None and name == "sgd":
        print("Using SGD: lr is: {}, L2 regularization is: {}".format(lr, config.l2))
        return optim.SGD(params, lr=lr, weight_decay=float(config.l2), momentum=config.momentum)
    elif config.optimizer.lower() == "sgd":
        print("Using SGD: lr is: {}, L2 regularization is: {}".format(lr, config.l2))
        return optim.SGD(params, lr=lr, weight_decay=float(config.l2), momentum=config.momentum)
    elif config.optimizer.lower() == "adam":
        print("Using Adam")
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        print("Illegal optimizer: {}".format(config.optimizer))
        exit(1)


def write_results(filename: str, insts):
    f = open(filename, 'w', encoding='utf-8')
    for inst in insts:
        for i in range(len(inst.input)):
            words = inst.input.words
            output = inst.output
            prediction = inst.prediction
            assert len(output) == len(prediction)
            f.write("{}\t{}\t{}\t{}\n".format(i, words[i], output[i], prediction[i]))
        f.write("\n")
    f.close()

