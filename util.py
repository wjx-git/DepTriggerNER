import torch
from collections import defaultdict


def remove_duplicates(features, labels, triggers, dataset):
    feature_dict = defaultdict(list)
    for feature, label, trigger in zip(features, labels, triggers):
        feature_dict[trigger].append((feature, label))

    for key, value in feature_dict.items():
        embedding = [v[0] for v in value]
        embedding = torch.mean(torch.stack(embedding), dim=0)
        for data in dataset:
            if key == data.trigger_key:
                data.trigger_vec = embedding

    duplicate_key = []
    for key, value in feature_dict.items():
        labels = [v[1] for v in value]
        labels = set(labels)
        if len(labels) > 1:
            duplicate_key.append(key)

    for key in duplicate_key:
        del feature_dict[key]

    for key, value in feature_dict.items():
        embedding = [f[0] for f in value]
        label = value[0][1]
        embedding = torch.mean(torch.stack(embedding), dim=0)
        feature_dict[key] = [embedding, label]

    trigger_key = []
    final_trigger = []
    for key, value in feature_dict.items():
        final_trigger.append(feature_dict[key][0])
        trigger_key.append(key)

    return torch.stack(final_trigger), trigger_key
