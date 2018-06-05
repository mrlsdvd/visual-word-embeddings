import torch
import numpy as np
import pickle
import os
import sys
import argparse
import pandas as pd
# Add path to config
sys.path.append('../../')
import config as conf
sys.path.append(conf.utils_path)
from vocabulary import Vocabulary

# VOCAB_LOAD_PATH = os.path.join(conf.models_path, 'topdown', 'infos_td-best.pkl')
VOCAB_PATH = os.path.join(conf.models_path, 'topdown', 'vocab.pkl')
EMBEDDINGS_PATH = os.path.join(conf.models_path, 'topdown', 'model-best.pth')

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
    embeddings = torch.load(embeddings_path, map_location='cpu')['embed.0.weight']

# ------Load the vocabulary from downloaded pretrained model------------
    # # Create a vocab wrapper and add some special tokens.
    # with open(VOCAB_LOAD_PATH, 'rb') as f:
    #     infos = pickle.load(f, encoding='latin1')
    # loaded_vocab = infos['vocab']

    # vocab = Vocabulary()
    # vocab.add_word('<pad>')
    # vocab.add_word('<start>')
    # vocab.add_word('<end>')
    # vocab.add_word('<unk>')

    
    # # Add the words to the vocabulary.
    # for i, word in enumerate(loaded_vocab.values()):
    #     vocab.add_word(word)

    # print(vocab.word2idx.keys())
    # # Save vocabulary wrapper
    # with open(vocab_path, 'wb') as f:
    #    pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
# ----------------------------------------------------------------------

    if embed_type == 'numpy':
        embeddings = embeddings.detach().numpy()

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
