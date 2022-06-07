from config import ContextEmb, batching_list_instances
from config.eval import evaluate_batch_insts
from config.utils import get_optimizer
from model.linear_crf_inferencer import LinearCRF
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from model.base_encoder import Encoder
from model.attention import Attention


class NEREncoder(nn.Module):
    def __init__(self, config, trigger_encoder):
        super(NEREncoder, self).__init__()
        self.config = config
        self.device = config.device
        self.base_encoder = Encoder(config)

        # 共享基础编码器
        # self.base_encoder = trigger_encoder.base_encoder
        self.match_encoder = trigger_encoder.base_encoder
        self.match_attention = trigger_encoder.attention
        # self.match_attention = Attention(config)
        self.label_size = config.label_size
        self.inferencer = LinearCRF(config)
        self.hidden2tag = nn.Linear(config.hidden_dim * 2, self.label_size).to(self.device)

        self.w1 = nn.Linear(config.hidden_dim, config.hidden_dim // 2).to(self.device)
        self.w2 = nn.Linear(config.hidden_dim // 2, config.hidden_dim // 2).to(self.device)
        self.attn1 = nn.Linear(config.hidden_dim // 2, 1).to(self.device)
        self.attn2 = nn.Linear(config.hidden_dim + config.hidden_dim // 2, 1).to(self.device)
        self.attn3 = nn.Linear(config.hidden_dim // 2, 1).to(self.device)
        self.tanh = nn.Tanh().to(self.device)

    def forward(self, word_seq_tensor: torch.Tensor,
                word_seq_lens: torch.Tensor,
                batch_context_emb: torch.Tensor,
                char_inputs: torch.Tensor,
                char_seq_lens: torch.Tensor,
                trigger_position,
                tags):

        batch_size = word_seq_tensor.size(0)
        max_sent_len = word_seq_tensor.size(1)

        #
        output, sentence_mask, _, _ = self.base_encoder(word_seq_tensor,
                                                        word_seq_lens,
                                                        batch_context_emb,
                                                        char_inputs,
                                                        char_seq_lens,
                                                        trigger_position)
        _, _, trigger_vec, trigger_mask = self.match_encoder(word_seq_tensor,
                                                             word_seq_lens,
                                                             batch_context_emb,
                                                             char_inputs,
                                                             char_seq_lens,
                                                             trigger_position
                                                             )
        weights = []
        if trigger_vec is not None:
            trig_rep_m = self.match_attention.attention(trigger_vec, trigger_mask)
            trig_rep = trig_rep_m.detach()
            for i in range(len(output)):
                trig_applied = self.tanh(
                    self.w1(output[i].unsqueeze(0)) + self.w2(trig_rep[i].unsqueeze(0).unsqueeze(0))
                )
                x = self.attn1(trig_applied)
                x = torch.mul(x.squeeze(0), sentence_mask[i].unsqueeze(1))
                x[x == 0] = float('-inf')
                weights.append(x)
        else:
            for i in range(len(output)):
                trig_applied = self.tanh(
                    self.w1(output[i].unsqueeze(0)) + self.w1(output[i].unsqueeze(0))
                )
                x = self.attn1(trig_applied)  # 63,1
                x = torch.mul(x.squeeze(0), sentence_mask[i].unsqueeze(1))
                x[x == 0] = float('-inf')
                weights.append(x)
        normalized_weights = F.softmax(torch.stack(weights), 1)
        attn_applied1 = torch.mul(normalized_weights.repeat(1, 1, output.size(2)), output)

        output = torch.cat([output, attn_applied1], dim=2)  # short cut
        lstm_scores = self.hidden2tag(output)
        maskTemp = torch.arange(
            1,
            max_sent_len + 1,
            dtype=torch.long
        ).view(1, max_sent_len).expand(batch_size, max_sent_len).to(self.device)
        mask = torch.le(maskTemp, word_seq_lens.view(batch_size, 1).expand(batch_size, max_sent_len)).to(self.device)

        if self.inferencer is not None:
            unlabeled_score, labeled_score = self.inferencer(lstm_scores, word_seq_lens, tags, mask)
            sequence_loss = unlabeled_score - labeled_score
            return sequence_loss
        sequence_loss = self.compute_nll_loss(lstm_scores, tags, mask, word_seq_lens)
        return sequence_loss

    def decode(self, word_seq_tensor: torch.Tensor,
               word_seq_lens: torch.Tensor,
               batch_context_emb: torch.Tensor,
               char_inputs: torch.Tensor,
               char_seq_lens: torch.Tensor,
               trig_rep,
               eval=None):

        output, sentence_mask, _, _ = self.base_encoder(word_seq_tensor,
                                                        word_seq_lens,
                                                        batch_context_emb,
                                                        char_inputs,
                                                        char_seq_lens,
                                                        None)
        match_output, match_sentence_mask, _, _ = self.match_encoder(word_seq_tensor,
                                                                     word_seq_lens,
                                                                     batch_context_emb,
                                                                     char_inputs,
                                                                     char_seq_lens,
                                                                     None)
        sent_rep = self.match_attention.attention(match_output, match_sentence_mask)

        trig_vec = trig_rep[0]

        n = sent_rep.size(0)
        m = trig_vec.size(0)
        d = sent_rep.size(1)

        sent_rep_dist = sent_rep.unsqueeze(1).expand(n, m, d)
        trig_vec_dist = trig_vec.unsqueeze(0).expand(n, m, d)

        dist = torch.pow(sent_rep_dist - trig_vec_dist, 2).sum(2).sqrt()
        #
        dvalue, dindices = torch.min(dist, dim=1)
        trigger_list = [trig_vec[i] for i in dindices.tolist()]

        # if eval:
        #     dvalue, dindices = torch.topk(dist, k=3, dim=1, largest=False)
        #     trigger_list = []
        #     for j in dindices.tolist():
        #        temp = []
        #        for i in j:
        #            temp.append(trig_vec[i].tolist())
        #        temp = torch.FloatTensor(temp)
        #        trigger_list.append(torch.mean(temp, dim=0))

        trig_rep = torch.stack(trigger_list)
        # attention
        weights = []
        for i in range(len(output)):
            trig_applied = self.tanh(self.w1(output[i].unsqueeze(0)) + self.w2(trig_rep[i].unsqueeze(0).unsqueeze(0)))
            x = self.attn1(trig_applied)
            x = torch.mul(x.squeeze(0), sentence_mask[i].unsqueeze(1))
            x[x == 0] = float('-inf')
            weights.append(x)
        normalized_weights = F.softmax(torch.stack(weights), 1)
        attn_applied1 = torch.mul(normalized_weights.repeat(1, 1, output.size(2)), output)

        output = torch.cat([output, attn_applied1], dim=2)
        # output = torch.add(attn_applied1, self.w3(output))
        lstm_scores = self.hidden2tag(output)
        bestScores, decodeIdx = self.inferencer.decode(lstm_scores, word_seq_lens, None)

        return bestScores, decodeIdx


class NEREncoderTrainer(object):
    def __init__(self, model, config, dev, test, train, triggers):
        self.model = model
        self.config = config
        self.device = config.device
        self.input_size = config.embedding_dim
        self.context_emb = config.context_emb
        self.use_char = config.use_char_rnn
        self.triggers = triggers
        if self.context_emb:
            self.input_size += config.context_emb_size
        if self.use_char:
            self.input_size += config.charlstm_hidden_dim
        self.optimizer = get_optimizer(config.ner_learning_rate, config.weight_decay,
                                       config, self.model, config.ner_optimizer)
        self.dev = dev
        self.test = test
        self.train = train

    def train_model(self, num_epochs, train_data, eval):
        batched_data = batching_list_instances(self.config, train_data)
        # train_batches = batching_list_instances(self.config, self.train)
        dev_batches = batching_list_instances(self.config, self.dev)
        test_batches = batching_list_instances(self.config, self.test)
        for epoch in range(num_epochs):
            epoch_loss = 0
            self.model.zero_grad()
            for batch in tqdm(batched_data):
                self.model.train()
                loss = self.model(*batch[0:5], batch[-2], batch[-3])
                epoch_loss += loss.data
                loss.backward()
                self.optimizer.step()
                self.model.zero_grad()
            print('epoch: {}, train, loss: {}'.format(epoch, epoch_loss))
            if eval:
                self.model.eval()
                # self.evaluate_model(dev_batches, "dev", self.dev, self.triggers)
                self.evaluate_model(test_batches, "test", self.test, self.triggers)
                self.model.zero_grad()
        # self.evaluate_model(dev_batches, "dev", self.dev, self.triggers)
        return self.model

    def evaluate_model(self, batch_insts_ids, name, insts, triggers):
        ## evaluation
        metrics = np.asarray([0, 0, 0], dtype=int)
        batch_id = 0
        batch_size = self.config.batch_size
        for batch in batch_insts_ids:
            one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
            batch_max_scores, batch_max_ids = self.model.decode(*batch[0:5], triggers, eval=name)
            metrics += evaluate_batch_insts(one_batch_insts, batch_max_ids, batch[6], batch[1], self.config.idx2labels,
                                            self.config.use_crf_layer)
            batch_id += 1
        p, total_predict, total_entity = metrics[0], metrics[1], metrics[2]
        precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
        recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
        fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
        print("[%s set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, precision, recall, fscore), flush=True)
