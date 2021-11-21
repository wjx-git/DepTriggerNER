import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Union
from common import Instance
import torch
from enum import Enum
import os

START = "<START>"
STOP = "<STOP>"
PAD = "<PAD>"
UNK = "<UNK>"

class ContextEmb(Enum):
    none = 0
    elmo = 1
    bert = 2 # not support yet
    flair = 3 # not support yet


class Config:
    def __init__(self, args) -> None:
        """
        Construct the arguments and some hyperparameters
        :param args:
        """

        # Predefined label string.
        self.PAD = PAD
        self.B = "B-"
        self.I = "I-"
        self.S = "S-"
        self.E = "E-"
        self.O = "O"
        self.START_TAG = START
        self.STOP_TAG = STOP
        self.UNK = UNK
        self.unk_id = -1

        # Model hyper parameters
        self.embedding_file = args.embedding_file
        self.embedding_dim = args.embedding_dim
        self.context_emb = args.context_emb
        self.context_emb_size = 256
        self.embedding, self.embedding_dim = self.read_pretrain_embedding()
        self.word_embedding = None
        self.seed = args.seed
        self.digit2zero = args.digit2zero
        self.hidden_dim = args.hidden_dim
        self.rep_hidden_dim = args.hidden_dim
        self.use_brnn = True
        self.num_layers = 1
        self.dropout = args.dropout
        self.char_emb_size = 25
        self.charlstm_hidden_dim = 50
        self.use_char_rnn = args.use_char_rnn
        self.use_crf_layer = args.use_crf_layer

        # Data specification
        self.percentage = args.percentage
        self.dataset = args.dataset
        self.train_file = "dataset/" + self.dataset + "/train_20.txt"
        self.train_all_file = "dataset/" + self.dataset + "/train.txt"
        if self.dataset == "Laptop-reviews":
            self.dev_file = "dataset/" + self.dataset + "/test.txt"
        else:
            self.dev_file = "dataset/" + self.dataset + "/dev.txt"
        self.test_file = "dataset/" + self.dataset + "/test.txt"
        self.trigger_file = "dataset/" + self.dataset + "/trigger_prim_100.txt"
        self.label2idx = {}
        self.idx2labels = []
        self.char2idx = {}
        self.idx2char = []
        self.num_char = 0
        self.train_num = args.train_num
        self.dev_num = args.dev_num
        self.test_num = args.test_num

        # Training hyperparameter
        self.model_folder = args.model_folder
        self.ner_optimizer = args.ner_optimizer.lower()
        self.weight_decay = args.weight_decay
        self.trig_optimizer = args.trig_optimizer.lower()
        self.ner_learning_rate = args.ner_learning_rate
        self.tri_learning_rate = args.trig_learning_rate
        self.momentum = args.momentum
        self.l2 = args.l2
        self.num_epochs = args.num_epochs
        self.num_epochs_soft = args.num_epochs_soft
        self.use_dev = True
        self.batch_size = args.batch_size
        self.clip = 5
        self.lr_decay = args.lr_decay
        self.device = torch.device(args.device)
        self.ds_setting = args.ds_setting
        print(self.ds_setting)

    def read_pretrain_embedding(self) -> Tuple[Union[Dict[str, np.array], None], int]:
        """
        Read the pretrained word embeddings, return the complete embeddings and the embedding dimension
        :return:
        """
        print("reading the pretraing embedding: %s" % (self.embedding_file))
        if self.embedding_file is None:
            print("pretrain embedding in None, using random embedding")
            return None, self.embedding_dim
        else:
            exists = os.path.isfile(self.embedding_file)
            if not exists:
                print("[Warning] pretrain embedding file not exists, using random embedding")
                return None, self.embedding_dim
        embedding_dim = -1
        embedding = {}
        with open(self.embedding_file, 'r', encoding='utf-8') as file:
            for line in tqdm(file.readlines()):
                tokens = line.strip().split()
                if embedding_dim < 0:
                    embedding_dim = len(tokens) - 1
                else:
                    assert (embedding_dim == len(tokens) - 1)
                embedd = np.empty([1, embedding_dim])
                embedd[:] = tokens[1:]
                embedding[tokens[0]] = embedd
        return embedding, embedding_dim

    def build_word_idx(self, train_insts: List[Instance], dev_insts: List[Instance] = None, test_insts: List[Instance] = None) -> None:
        """
        Build the vocab 2 idx for all instances
        :param train_insts:
        :param dev_insts:
        :param test_insts:
        :return:
        """
        self.word2idx = {}
        self.idx2word = []
        self.word2idx[self.PAD] = 0
        self.idx2word.append(self.PAD)
        self.word2idx[self.UNK] = 1
        self.unk_id = 1
        self.idx2word.append(self.UNK)

        self.char2idx[self.PAD] = 0
        self.idx2char.append(self.PAD)
        self.char2idx[self.UNK] = 1
        self.idx2char.append(self.UNK)

        # extract char on train, dev, test. For word embedding, we should only need train dataset
        if dev_insts is not None and test_insts is not None:
            whole_sets = train_insts + dev_insts + test_insts
        else:
            whole_sets = train_insts

        for inst in whole_sets:
            for word in inst.input.words:
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word.append(word)
        # extract char only on train (doesn't matter for dev and test)
        for inst in train_insts:
            for word in inst.input.words:
                for c in word:
                    if c not in self.char2idx:
                        self.char2idx[c] = len(self.idx2char)
                        self.idx2char.append(c)
        self.num_char = len(self.idx2char)

    def add_word_idx(self, match_insts):
        for inst in match_insts:
            for word in inst.input.words:
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word.append(word)

    def build_emb_table(self) -> None:
        """
        build the embedding table with pretrained word embeddings (if given otherwise, use random embeddings)
        :return:
        """
        print("Building the embedding table for vocabulary...")
        scale = np.sqrt(3.0 / self.embedding_dim)
        if self.embedding is not None:
            print("[Info] Use the pretrained word embedding to initialize: %d x %d" % (len(self.word2idx), self.embedding_dim))
            self.word_embedding = np.empty([len(self.word2idx), self.embedding_dim])
            for word in self.word2idx:
                if word in self.embedding:
                    self.word_embedding[self.word2idx[word], :] = self.embedding[word]
                elif word.lower() in self.embedding:
                    self.word_embedding[self.word2idx[word], :] = self.embedding[word.lower()]
                else:
                    # self.word_embedding[self.word2idx[word], :] = self.embedding[self.UNK]
                    self.word_embedding[self.word2idx[word], :] = np.random.uniform(-scale, scale, [1, self.embedding_dim])
        else:
            self.word_embedding = np.empty([len(self.word2idx), self.embedding_dim])
            for word in self.word2idx:
                self.word_embedding[self.word2idx[word], :] = np.random.uniform(-scale, scale, [1, self.embedding_dim])

    def build_label_idx(self, insts: List[Instance]) -> None:
        """
        Build the mapping from label to index and index to labels.
        :param insts: list of instances.
        :return:
        """
        #self.triggerlabel = {}
        self.label2idx[self.PAD] = len(self.label2idx)
        self.idx2labels.append(self.PAD)
        for inst in insts:
            for label in inst.output:
                if label not in self.label2idx:
                    self.idx2labels.append(label)
                    self.label2idx[label] = len(self.label2idx)
                #if label not in [START,STOP,PAD,UNK,'O','T'] and label.split('-')[1] not in self.triggerlabel:
                #    self.triggerlabel[label.split('-')[1]] = len(self.triggerlabel)

        self.label2idx[self.START_TAG] = len(self.label2idx)
        self.idx2labels.append(self.START_TAG)
        self.label2idx[self.STOP_TAG] = len(self.label2idx)
        self.idx2labels.append(self.STOP_TAG)
        self.label_size = len(self.label2idx)
        #self.trig_label_size = len(self.triggerlabel)
        self.start_label_id = self.label2idx[self.START_TAG]
        self.stop_label_id = self.label2idx[self.STOP_TAG]
        print("#labels: {}".format(self.label_size))
        print("label2idx: {}".format(self.label2idx))

    def use_iobes(self, insts: List[Instance]) -> None:
        """
        Use IOBES tagging schema to replace the IOB tagging schema in the instance
        :param insts:
        :return:
        """
        for inst in insts:
            output = inst.output
            for pos in range(len(inst)):
                curr_entity = output[pos]
                if pos == len(inst) - 1:
                    if curr_entity.startswith(self.B):
                        output[pos] = curr_entity.replace(self.B, self.S)
                    elif curr_entity.startswith(self.I):
                        output[pos] = curr_entity.replace(self.I, self.E)
                else:
                    next_entity = output[pos + 1]
                    if curr_entity.startswith(self.B):
                        if next_entity.startswith(self.O) or next_entity.startswith(self.B):
                            output[pos] = curr_entity.replace(self.B, self.S)
                    elif curr_entity.startswith(self.I):
                        if next_entity.startswith(self.O) or next_entity.startswith(self.B):
                            output[pos] = curr_entity.replace(self.I, self.E)

    def map_insts_ids(self, insts: List[Instance]):
        """
        Create id for word, char and label in each instance.
        :param insts:
        :return:
        """
        for inst in insts:
            words = inst.input.words
            inst.word_ids = []
            inst.char_ids = []
            inst.output_ids = [] if inst.output else None
            for word in words:
                inst.word_ids.append(self.word2idx.get(word, self.unk_id))
                char_id = [self.char2idx.get(c, self.unk_id) for c in word]
                inst.char_ids.append(char_id)
            if inst.output:
                for label in inst.output:
                    inst.output_ids.append(self.label2idx.get(label, self.label2idx['O']))
