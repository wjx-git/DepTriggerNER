from stanfordcorenlp import StanfordCoreNLP
# from nltk.tree import Tree


class Parser:
    def __init__(self):
        path = r'D:\Project\stanford-corenlp-4.2.0'
        self.parser = StanfordCoreNLP(path)

    def parse_sentence(self, sentence):
        deprel = self.parser.dependency_parse(sentence)
        words = self.parser.word_tokenize(sentence)
        postag = self.parser.pos_tag(sentence)
        return deprel, words, postag

    def close(self):
        self.parser.close()


if __name__ == '__main__':
    # from stanfordcorenlp import StanfordCoreNLP

    path = r'D:\Project\stanford-corenlp-4.2.0'
    parser = StanfordCoreNLP(path)
    # sentence = r'EU rejects German call to boycott British lamb.'
    # sentence = r'John is the leader of out group'
    # deprel = parser.dependency_parse(sentence)
    # words = parser.word_tokenize(sentence)
    # postag = parser.pos_tag(sentence)
    # print(deprel)
    # print(words)
    # print(postag)

    # tree = Tree.fromstring(parser.dependency_parse(sentence))
    # tree.draw()
    # parser.close()
