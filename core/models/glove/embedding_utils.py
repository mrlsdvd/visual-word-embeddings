import torch
import numpy as np
import pickle
import os
import sys
import argparse
import pandas as pd
import csv
# Add path to config
sys.path.append('../../')
import config as conf
sys.path.append(conf.utils_path)
from vocabulary import Vocabulary

VOCAB_PATH = os.path.join(conf.models_path, 'glove', 'vocab.pkl')
# VOCAB_PATH = os.path.join(conf.core_models_path, 'bimodal_autoencoder', 'data', 'vocab.pkl')
EMBEDDINGS_PATH = os.path.join(conf.models_path, 'glove', 'glove.6B', 'glove.6B.200d.txt')

def get_embeddings(vocab_path=VOCAB_PATH, embeddings_path=EMBEDDINGS_PATH, embed_type='numpy'):
    """Load saved trained embeddings and corresponding vocabulary

    Args:
        vocab_path (str): path to vocabulary pkl file
        embedding_path (str): path to pretrained glove embeddings file
        embed_type (str): whether to return embedding matrix as 'numpy' array or
                         as torch 'tensor'
    """
    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    # Load the trained model parameters
    embeddings = pd.read_table(EMBEDDINGS_PATH, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE, encoding='utf-8')
    # # embeddings = embeddings.dot(embeddings.transpose())
    vocab_index = embeddings.index.tolist()


# ------Load the vocabulary from downloaded pretrained model------------
    # # Create a vocab wrapper and add some special tokens.
    # vocab = Vocabulary()
    #
    # # Add the words to the vocabulary.
    # for i, word in enumerate(vocab_index):
    #     vocab.add_word(word)
    #
    # vocab.add_word('<pad>')
    # vocab.add_word('<start>')
    # vocab.add_word('<end>')
    # vocab.add_word('<unk>')
    #
    # # Save vocabulary wrapper
    # with open(vocab_path, 'wb') as f:
    #    pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
# ----------------------------------------------------------------------
    if embed_type == 'numpy':
        embeddings = embeddings.values

    # Add special token vecotors to embedding matrix
    special_embeddings = np.random.rand(4, embeddings.shape[1])
    embeddings = np.vstack([embeddings, special_embeddings])

    print(embeddings.shape)


    return embeddings, vocab


def main(args):
    embeddings, vocab = get_embeddings(vocab_path=args.vocab_path,
                                       embeddings_path=args.embeddings_path,
                                       embed_type=args.embed_type)
    # Check embedding shape
    print(embeddings.shape)
    # Check vocabulary
    print(len(vocab))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_path', type=str, default=EMBEDDINGS_PATH, help='path for pretrained glove embeddings')
    parser.add_argument('--vocab_path', type=str, default=VOCAB_PATH, help='path for vocabulary wrapper')
    parser.add_argument('--embed_type', type=str, default='numpy', help='type of embedding matrix')
    args = parser.parse_args()
    main(args)
