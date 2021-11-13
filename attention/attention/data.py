import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import time as t

class AttentionDataset(Dataset):
    def __init__(self, input_sentences, output_sentences, target_sentences, max_len):
        self.data = []
        self.word2id = {}
        self.id2word = {}
        word_list = []
        sentence_num = len(input_sentences)
        for i in range(0, sentence_num):
            input_sentence, output_sentence, target_sentence = input_sentences[i], output_sentences[i], target_sentences[i]
            input_sentence = input_sentence + (max_len - len(input_sentence.split())) * " <pad>"
            output_sentence = "<st> " + output_sentence + (max_len - 1 - len(output_sentence.split())) * " <pad>"
            target_sentence = target_sentence + " <end>" + (max_len - 1 - len(target_sentence.split())) * " <pad>"

            # input_sentence = input_sentence
            # output_sentence = "<st> " + output_sentence
            # target_sentence = target_sentence + " <end>"

            self.data.extend([(input_sentence, output_sentence, target_sentence)])
            sentences = [input_sentence, output_sentence, target_sentence]
            word_list += " ".join(sentences).split()

        word_list = list(set(word.strip('.').strip(',') for word in word_list))
        self.word2id = {w: i for i, w in enumerate(word_list)}
        self.id2word = [w for i, w in enumerate(word_list)]
        # print(self.word2id)
        # print(self.id2word)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]

    def get_word2id(self):
        return self.word2id
    
    def get_id2word(self):
        return self.id2word

def get_loader(dataset, batch_size, shuffle=True):
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return data_loader

def make_batch(input_batch, output_batch, target_batch, vocab_size, word2id):
    inputs = []
    outputs = []
    targets = []
    # masks = []
    for i in range(0, len(input_batch)):
        inputt, output, target = input_batch[i], output_batch[i], target_batch[i]
        # length = len(inputt.split())
        # masks.append([1]*length + [0]*(max_len-length))
        inputs.append(np.eye(vocab_size)[[word2id[word.strip('.').strip(',')] for word in inputt.split()]])
        outputs.append(np.eye(vocab_size)[[word2id[word.strip('.').strip(',')] for word in output.split()]])
        targets.append([word2id[word.strip('.').strip(',')] for word in target.split()])
    return torch.FloatTensor(inputs), torch.FloatTensor(outputs), torch.tensor(targets)
