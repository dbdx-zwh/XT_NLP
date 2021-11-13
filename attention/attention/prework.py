import re

def read_data():
    input_sentences, output_sentences, target_sentences = [], [], []
    f = open('/Users/dbdxzwh/Desktop/train.txt', 'r')
    f1 = open('en.txt', 'w')
    f2 = open('fr.txt', 'w')
    lines = f.readlines()
    for line in lines:
        sentences = line.split('.', 1)
        if (len(sentences) == 1):
            sentences = line.split('!', 1)
            if (len(sentences) == 1):
                sentences = line.split('?', 1)
        assert (len(sentences) == 2)
        sentences[1] = re.sub(u'\u202f', "", sentences[1])
        f1.write(sentences[0]+'\n')
        f2.write(sentences[1])

read_data()