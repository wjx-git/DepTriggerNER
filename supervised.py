from model.trigger_encoder import *
from model.ner_encoder import *
from model.base_encoder import Encoder
from config import Reader, Config
import argparse
import random


def parse_arguments(parser):
    parser.add_argument('--device', type=str, default="cpu", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2','cuda:3', 'cuda:4', 'cuda:5', 'cuda:6'],
                        help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--digit2zero', action="store_true", default=True,
                        help="convert the number to 0, make it true is better")
    parser.add_argument('--dataset', type=str, default="CONLL")
    parser.add_argument('--embedding_file', type=str,
                        # default=None,
                        default=r"gloveEN/glove.6B.100d.txt",
                        help="we will using random embeddings if file do not exist")
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--trig_optimizer', type=str, default="adam")
    parser.add_argument('--ner_optimizer', type=str, default="sgd")
    parser.add_argument('--trig_learning_rate', type=float, default=1e-3)
    parser.add_argument('--ner_learning_rate', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=10, help="default batch size is 10 (works well)")
    parser.add_argument('--num_epochs', type=int, default=20, help="Usually we set to 10~20.")
    parser.add_argument('--num_epochs_soft', type=int, default=10, help="Usually we set to 10~20.")
    parser.add_argument('--train_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--dev_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--test_num', type=int, default=-1, help="-1 means all the data")
    # model hyperparameter
    parser.add_argument('--model_folder', type=str, default="english_model", help="The name to save the model files")
    parser.add_argument('--hidden_dim', type=int, default=200, help="hidden size of the LSTM, 128 or 200")
    parser.add_argument('--use_crf_layer', type=int, default=1, help="1 is for using crf layer, 0 for not using CRF layer", choices=[0,1])
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout for embedding")
    parser.add_argument('--use_char_rnn', type=int, default=1, choices=[0, 1], help="use character-level lstm, 0 or 1")
    parser.add_argument('--context_emb', type=bool, default=False, help="contextual word embedding")
    parser.add_argument('--ds_setting', nargs='+', help="+ hard / soft matching")  # soft, hard
    parser.add_argument('--percentage', type=int, default=10, help="how much percentage training dataset to use")

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


parser = argparse.ArgumentParser()
opt = parse_arguments(parser)
conf = Config(opt)
reader = Reader(conf.digit2zero)

dataset, max_length, label_length = reader.read_trigger_txt(conf.trigger_file, -1)
reader.merge_labels(dataset)

original_train = reader.read_txt(conf.train_all_file, conf.train_num)
devs = reader.read_txt(conf.dev_file, conf.dev_num)
tests = reader.read_txt(conf.test_file, conf.test_num)

# setting for data
conf.use_iobes(dataset)
conf.use_iobes(original_train)
conf.use_iobes(devs)
conf.use_iobes(tests)

conf.optimizer = opt.trig_optimizer
conf.build_label_idx(dataset)
conf.build_word_idx(dataset, devs, tests)
conf.build_emb_table()
conf.map_insts_ids(dataset)
conf.map_insts_ids(original_train)
conf.map_insts_ids(devs)
conf.map_insts_ids(tests)

base_encoder = Encoder(conf)

# matching module training
dataset = reader.trigger_percentage(dataset, conf.percentage)
random.shuffle(dataset)
print('total sentences for train: {}'.format(len(dataset)))

trigger_encoder = TriggerEncoder(conf, base_encoder, label_length)
trigger_trainer = TriggerEncoderTrainer(trigger_encoder, conf, devs, tests)
trigger_trainer.train_model(conf.num_epochs_soft, dataset)

logits, triggers = trigger_trainer.get_triggervec(dataset)
triggers_remove = trigger_trainer.remove_duplicates(logits, triggers, dataset)

# sequence labeling module training
random.shuffle(dataset)
ner_encoder = NEREncoder(conf, trigger_encoder)
ner_trainer = NEREncoderTrainer(ner_encoder, conf, devs, tests, original_train, triggers_remove)
ner_trainer.train_model(conf.num_epochs, dataset, eval=True)
