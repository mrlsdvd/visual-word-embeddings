import nltk
import pickle
import argparse
from collections import Counter
from pycocotools.coco import COCO
import csv
import sys, os
import pandas as pd
# Add path to config
sys.path.append('../../')
import config as conf


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not str(word) in self.word2idx:
            # word = str(word) + '_{}'.format(self.idx)
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
        self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return self.idx

def build_vocab(embeddings_path):
    """Build a simple vocabulary wrapper from embeddings

    Args:
        embedding_path (str): path to pretrained glove embeddings file
    """
    # Load the trained model parameters
    embeddings = pd.read_table(embeddings_path, sep="\s+", index_col=0, header=None, quoting=csv.QUOTE_NONE, encoding='utf-8')
    print("Total embeddings size: {}".format(embeddings.shape))
    vocab_index = embeddings.index.tolist()
    # print(vocab_index)

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()

    # Add the words to the vocabulary.
    for i, word in enumerate(vocab_index):
        vocab.add_word(str(word))

    # Add special words
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    return vocab

def main(args):
    vocab_path = args.vocab_path
    embeddings_path = args.embeddings_path

    vocab = build_vocab(embeddings_path=embeddings_path)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--embeddings_path', type=str, default=os.path.join(conf.models_path, 'glove', 'glove.6B', 'glove.6B.200d.txt'),
                        help='path for loading pretrained embeddings')
    args = parser.parse_args()
    main(args)
