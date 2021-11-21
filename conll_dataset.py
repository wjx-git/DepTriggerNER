from tqdm import tqdm

from common.stanfordParser import Parser
# from collections import defaultdict

remove = ['ROOT', 'punct']
sp = Parser()


def clean_dataset_without_trigger(train_data):
    data_trigger = []
    for sent, label in train_data:
        if 'T-' in ' '.join(label):
            data_trigger.append([sent, label])
    print('dataset size：{}'.format(len(data_trigger)))
    return data_trigger


def sample_per_entity(file, outp):
    dataset = load_dataset(file)
    new_dataset = []
    counts = 0
    for _, labels in tqdm(dataset):
        for label in labels:
            if label.startswith('B'):
                counts += 1

    for sentence, labels in tqdm(dataset):
        tag = ["O"] * len(labels)
        if tag == labels:
            new_dataset.append([sentence, tag])
            continue
        flag = False
        for i, label in enumerate(labels):
            if label.startswith('B-') or label.startswith('I-'):
                tag[i] = label
                flag = True
            if label == 'O' and flag:
                new_dataset.append([sentence, tag])
                tag = ["O"] * len(labels)
                flag = False
                # break
    data = cat_data_label(new_dataset)
    print('{} saved dataset size：{}'.format(outp, len(data)))
    with open(outp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(data))


def load_dataset(file):
    train_data = []
    with open(file, 'r', encoding='utf-8') as f:
        sentence, labels = [], []
        for line in f.readlines():
            if line != '\n':
                line = line.strip().split()
                w, l = line[0], line[1]
                sentence.append(w)
                labels.append(l)
            else:
                train_data.append([sentence, labels])
                sentence, labels = [], []
    print('{} dataset size：{}'.format(file, len(train_data)))
    return train_data

def first_trigger(entity, deprel):
    first_stage = []
    for ind in entity:
        for dep in deprel:
            if dep[0] not in remove:
                if dep[1]-1 == ind and dep[2]-1 not in entity:
                    first_stage.append((dep[2]-1, dep[0]))
                if dep[2]-1 == ind and dep[1]-1 not in entity:
                    first_stage.append((dep[1]-1, dep[0]))
    return first_stage


def second_trigger(first_trigger, deprel, entity):
    second_stage = []
    for ind in first_trigger:
        for dep in deprel:
            if dep[0] not in remove:
                if ind[0] == dep[1]-1 and dep[2]-1 not in entity:
                    second_stage.append((dep[2]-1, ind[-1]))
                if ind[0] == dep[2]-1 and dep[1]-1 not in entity:
                    second_stage.append((dep[1]-1, ind[-1]))
    return second_stage


def merge_label(label1, label2):
    results = []
    for la1, la2 in zip(label1, label2):
        if la1 != 'O':
            results.append(la1)
        elif la2 != 'O':
            results.append(la2)
        else:
            results.append('O')
    return results


def clear_original_trigger(labels):
    res = [label if label.startswith('B') or label.startswith('I') else 'O' for label in labels]
    return res


def mark_sentence_trigger(sentence, labels):
    res = sp.parse_sentence(' '.join(sentence))
    deprel, words, postag = res[0], res[1], res[2]
    entity = []
    entity_type = None
    for i, tag in enumerate(labels):
        if tag.startswith('B-') or tag.startswith('I-'):
            entity_type = tag[2:]
            entity.append(i)
    if entity_type:
        first_stage = first_trigger(entity, deprel)
        second_stage = second_trigger(first_stage, deprel, entity)

        # first_label = ['O'] * len(words)
        # for label in first_stage:
        #     first_label[int(label[0])] = 'T-' + label[-1]

        second_label = ['O'] * len(words)
        for label in second_stage:
            second_label[int(label[0])] = 'T-' + label[-1]

        labels = clear_original_trigger(labels)
        # labels = merge_label(labels, first_label)
        labels = merge_label(labels, second_label)
    return labels


def cat_data_label(train_data):
    res = []
    for sentence, label in train_data:
        res.append(''.join([word + '\t' + label + '\n' for word, label in zip(sentence, label)]))
    return res


def generate_dataset_with_trigger(file, outp):
    original_data = load_dataset(file)
    dataset = []
    others = defaultdict(list)

    for d in tqdm(original_data):
        try:
            if len(d[0]) == 0:
                continue
            if 'B-' not in ' '.join(d[1]):
                dataset.append([d[0], d[1]])
                continue
            label = mark_sentence_trigger(d[0], d[1])
            dataset.append([d[0], label])
        except Exception as e:
            others[' '.join(d[0])].append(d[1])

    sp.close()
    no_trigger = []
    for k, labels in others.items():
        sent = k.split()
        tag = ['O'] * len(labels[0])
        for i in range(len(labels)):
            tag = merge_label(tag, labels[i])
        no_trigger.append([sent, tag])

    dataset = dataset + no_trigger
    print('no_trigger size: {}'.format(len(no_trigger)))

    res = cat_data_label(dataset)
    print('{} saved dataset size：{}'.format(outp, len(res)))
    with open(outp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(res))


if __name__ == '__main__':
    # infile = r'dataset/CONLL/train.txt'
    # outfile1 = r'dataset/CONLL/train_100.txt'
    # outfile2 = r'dataset/CONLL/trigger_sec_100.txt'
    # sample_per_entity(infile, outfile1)
    # generate_dataset_with_trigger(outfile1, outfile2)

    # infile = r'dataset/BC5CDR/train.txt'
    # outfile1 = r'dataset/BC5CDR/train_100.txt'
    # outfile2 = r'dataset/BC5CDR/trigger_sec_100.txt'
    # # sample_per_entity(infile, outfile1)
    # generate_dataset_with_trigger(outfile1, outfile2)

    # 统计每个实体关联的触发词数量
    # from collections import defaultdict, Counter
    #
    # bc5cdr_file = r'dataset/CONLL/trigger_20.txt'
    # data = load_dataset(bc5cdr_file)
    # counts = []
    # total = 0
    # for d in data:
    #     nums = 0
    #     total += 1
    #     for tag in d[1]:
    #         if tag.startswith('T-'):
    #             nums += 1
    #     counts.append(nums)
    # results = Counter(counts)
    # print(total)
    # print(results)

    # 统计句子长度分布
    from collections import defaultdict

    bc5cdr_file = r'dataset/BC5CDR/train.txt'
    data = load_dataset(bc5cdr_file)
    counts = defaultdict(int)
    for d in data:
        leng = len(d[0])
        if leng < 5:
            counts['1~5'] += 1
        elif 5 <= leng < 10:
            counts['5~10'] += 1
        elif 10 <= leng < 25:
            counts['10~25'] += 1
        elif 25 <= leng < 50:
            counts['25~50'] += 1
        elif 50 <= leng < 100:
            counts['50~100'] += 1
        elif 100 <= leng:
            counts['100~'] += 1
    # print(counts)
    for k, v in counts.items():
        print(k, round(v/4542, 4))

    # conll_ints = [4657, 1083, 2735, 4769, 4190, 3105, 2154, 1512, 726, 464, 323]
    # bc5cdr_ints = [0, 794, 943, 878, 656, 466, 279, 174, 105, 59, 53]
    # oth = 4471 - sum(bc5cdr_ints)
    # for i in bc5cdr_ints + [oth]:
    #     print(round(i/4471, 4))

    # import matplotlib.pyplot as plt
    # from pylab import *  # 支持中文
    #
    # mpl.rcParams['font.sans-serif'] = ['SimHei']
    #
    # names = ['0', '1', '2', '3', '4', '5', '6', '7', '>=8']
    # x = range(9)
    # CoNLL_ours = [0.177, 0.041, 0.13, 0.181, 0.159, 0.118, 0.082, 0.057, 0.083]
    # CoNLL_TriggerNER = [0, 0.178, 0.211, 0.196, 0.147, 0.104, 0.062, 0.039, 0.063]
    # BC5CDR_ours = [0.063, 0, 0.095, 0.101, 0.170, 0.166, 0.121, 0.095, 0.183]
    # BC5CDR_TriggerNER = [0, 0.126, 0.211, 0.214, 0.154, 0.098, 0.07, 0.04, 0.087]
    # plt.ylim(0, 0.25)  # 限定纵轴的范围
    # plt.plot(x, CoNLL_ours, marker='o', color='coral', linestyle='-', label=u'CoNLL-ours')
    # plt.plot(x, CoNLL_TriggerNER, marker='*', color='coral', linestyle='--', label=u'CoNLL-TriggerNER')
    # plt.plot(x, BC5CDR_ours, marker='o', color='blue', label=u'BC5CDR-ours')
    # plt.plot(x, BC5CDR_TriggerNER, marker='*', color='blue', linestyle='--', label=u'BC5CDR-TriggerNER')
    # plt.legend()  # 让图例生效
    # plt.xticks(x, names, rotation=45)
    # plt.margins(0)
    # plt.subplots_adjust(bottom=0.15)
    # plt.xlabel(u"The number of triggers of a entity has")  # X轴标签
    # plt.ylabel("percentage")  # Y轴标签
    #
    # plt.show()




