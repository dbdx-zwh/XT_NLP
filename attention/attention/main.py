# -*- coding: utf-8 -*-
import argparse
import numpy as np
from data import AttentionDataset, get_loader, make_batch
from model import Attention
import data
import tqdm
import time
import torch
import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser(description="Run attention model.")
    parser.add_argument('--en_path', type=str, default="en.txt")
    parser.add_argument('--fr_path', type=str, default="fr.txt")
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--epoches', type=int, default=1000)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--max_sentence_len', type=int, default=10)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=10)
    return parser.parse_args()

# input_sentences = ["I like apples", "I like apples"]
# output_sentences = ["I like apples.", "I like apples"]
# target_sentences = ["I like apples,", "I like apples"]

def read_data(path):
    f = open(path, 'r')
    lines = f.readlines()
    return lines

if __name__ == "__main__":
    args = parse_args()
    input_sentences = read_data(args.en_path)
    output_sentences = read_data(args.fr_path)
    target_sentences = read_data(args.fr_path)
    # time.sleep(10)
    dataset = AttentionDataset(input_sentences, output_sentences, target_sentences, args.max_sentence_len)
    vocab_size = len(dataset.id2word)
    word2id = dataset.get_word2id()
    dataloader = get_loader(dataset, args.batch_size)

    model = Attention(vocab_size, args)
    criterion = nn.CrossEntropyLoss(ignore_index=word2id["<pad>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.epoches):
        loss = 0.
        optimizer.zero_grad()
        for input_batch, output_batch, target_batch in dataloader:
            inputs, outputs, targets = make_batch(input_batch, output_batch, target_batch, vocab_size, word2id)
            hidden = torch.zeros(2, args.batch_size, args.hidden_size)
            predict = model(inputs, hidden, outputs)
            # print(predict.shape, targets.unsqueeze(1).shape)
            # time.sleep(10)
            for i in range(args.batch_size):
                # print(predict[i].shape, targets[i].unsqueeze(0).shape)
                loss += criterion(predict[i], targets[i])

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()