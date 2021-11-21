from config import ContextEmb, batching_list_instances
from config.utils import get_optimizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import Attention
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from collections import defaultdict


class ContrastiveLoss(nn.Module):
    def __init__(self, margin, device):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9
        self.device = device

    def forward(self, output1, output2, target, size_average=True):
        target = target.to(self.device)
        distances = (output2 - output1).pow(2).sum(1).to(self.device)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TriggerEncoder(nn.Module):
    def __init__(self, config, encoder, num_classes):
        super(TriggerEncoder, self).__init__()
        self.config = config
        self.device = config.device
        self.label_size = config.label_size

        self.base_encoder = encoder
        self.attention = Attention(self.config)
        # trigger classification layers
        self.trigger_type_layer = nn.Linear(config.hidden_dim // 2, num_classes).to(self.device)

        self.hidden2tag = nn.Linear(config.hidden_dim * 2, self.label_size).to(self.device)

        self.w1 = nn.Linear(config.hidden_dim, config.hidden_dim // 2).to(self.device)
        self.w2 = nn.Linear(config.hidden_dim // 2, config.hidden_dim // 2).to(self.device)
        self.attn1 = nn.Linear(config.hidden_dim // 2, 1).to(self.device)
        self.tanh = nn.Tanh().to(self.device)

    def forward(self, word_seq_tensor: torch.Tensor,
                word_seq_lens: torch.Tensor,
                batch_context_emb: torch.Tensor,
                char_inputs: torch.Tensor,
                char_seq_lens: torch.Tensor,
                trigger_position,
                trigger_label):
        output, sentence_mask, trigger_vec, trigger_mask = self.base_encoder(word_seq_tensor,
                                                                             word_seq_lens,
                                                                             batch_context_emb,
                                                                             char_inputs,
                                                                             char_seq_lens,
                                                                             trigger_position)
        trig_rep, sentence_vec_cat, trigger_vec_cat = self.attention(output,
                                                                     sentence_mask,
                                                                     trigger_vec,
                                                                     trigger_mask,
                                                                     trigger_label)
        # final_trigger_type = self.trigger_type_layer(trig_rep)  # [batch_size, num_classes]
        return trig_rep, sentence_vec_cat, trigger_vec_cat

    def test_tri_match(self, word_seq_tensor: torch.Tensor,
                       word_seq_lens: torch.Tensor,
                       batch_context_emb: torch.Tensor,
                       char_inputs: torch.Tensor,
                       char_seq_lens: torch.Tensor,
                       trigger_position,
                       trigger_label):
        output, sentence_mask, trigger_vec, trigger_mask = self.base_encoder(word_seq_tensor,
                                                                             word_seq_lens,
                                                                             batch_context_emb,
                                                                             char_inputs,
                                                                             char_seq_lens,
                                                                             trigger_position)
        trig_rep, sentence_vec_cat, trigger_vec_cat = self.attention(output,
                                                                     sentence_mask,
                                                                     trigger_vec,
                                                                     trigger_mask,
                                                                     trigger_label)
        # final_trigger_type = self.trigger_type_layer(trig_rep)  # [batch_size, num_classes]
        return trig_rep, sentence_vec_cat, trigger_vec_cat


class TriggerEncoderTrainer(object):
    """
    train SoftMatcher model
    """

    def __init__(self, model, config, dev, test):
        self.model = model
        self.config = config
        self.device = config.device
        self.contrastive_loss = ContrastiveLoss(1.0, self.device)
        self.dev = dev
        self.test = test
        # self.train = train
        self.optimizer = get_optimizer(config.tri_learning_rate, 0, config, self.model, config.trig_optimizer)

    def train_model(self, num_epochs, train_data):
        batched_data = batching_list_instances(self.config, train_data)  #
        for epoch in range(num_epochs):
            t_loss, m_loss = 0, 0
            self.model.zero_grad()
            for batch in tqdm(batched_data):
                self.model.train()
                trig_rep, match_trig, match_sent = self.model(*batch[0:5], batch[-2],batch[-1])

                # Semantic match loss
                soft_matching_loss = self.contrastive_loss(match_trig,
                                                           match_sent,
                                                           torch.stack([torch.tensor(1)] * trig_rep.size(0) +
                                                                       [torch.tensor(0)] * trig_rep.size(0)))

                m_loss += soft_matching_loss.data
                soft_matching_loss.backward(retain_graph=True)
                self.optimizer.step()
                self.model.zero_grad()
            print('epoch: {}, train, match loss: {}'.format(epoch, m_loss))
            self.test_model(batched_data)
            self.model.zero_grad()

    def test_model(self, batched_data):
        self.model.eval()
        match_target_list = []
        matched_list = []
        for batch in tqdm(batched_data):
            trig_rep, match_trig, match_sent = self.model.test_tri_match(*batch[0:5], batch[-2], batch[-1])

            match_target_list.extend([torch.tensor(1)] * trig_rep.size(0) + [torch.tensor(0)] * trig_rep.size(0))
            distances = (match_trig - match_sent).pow(2).sum(1)
            distances = torch.sqrt(distances)
            matched_list.extend((distances < 1.0).long().tolist())

        print("soft matching accuracy ", accuracy_score(matched_list, match_target_list))

    def get_triggervec(self, data):
        batched_data = batching_list_instances(self.config, data)
        self.model.eval()
        logits_list = []
        # predicted_list = []
        trigger_list = []
        for index, batch in enumerate(batched_data):
            trig_rep, _, _ = self.model.test_tri_match(*batch[0:5], batch[-2], batch[-1])
            # trig_type_value, trig_type_predicted = torch.max(trig_type_probas, 1)  # return: max value, max value index
            ne_batch_insts = data[index * self.config.batch_size:(index + 1) * self.config.batch_size]
            for idx in range(len(trig_rep)):
                ne_batch_insts[idx].trigger_vec = trig_rep[idx]
            logits_list.extend(trig_rep)
            word_seq = batch[-2]
            trigger_list.extend(
                [" ".join(self.config.idx2word[index] for index in indices if index != 0) for indices in word_seq]
            )

        return logits_list, trigger_list

    def remove_duplicates(self, features, triggers, dataset):
        feature_dict = defaultdict(list)
        for feature, trigger in zip(features, triggers):
            feature_dict[trigger].append(feature)

        for key, value in feature_dict.items():
            embedding = [v for v in value]
            embedding = torch.mean(torch.stack(embedding), dim=0)
            for data in dataset:
                if key == data.trigger_key:
                    data.trigger_vec = embedding

        trigger_key = []
        final_trigger = []
        for key, value in feature_dict.items():
            final_trigger.append(torch.mean(torch.stack(value), dim=0))
            trigger_key.append(key)

        return torch.stack(final_trigger), trigger_key
