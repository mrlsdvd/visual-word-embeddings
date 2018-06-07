import torch
import numpy as np
import pickle
import os
import sys
import argparse
# Add path to config
sys.path.append('../../')
import config as conf
sys.path.append(conf.utils_path)
from vocabulary import Vocabulary

VOCAB_PATH = os.path.join(conf.models_path, 'image_captioning', 'vocab.pkl')
DECODER_PATH = os.path.join(conf.models_path, 'glove_image_captioning', 'decoder-1-300.ckpt')

def get_embeddings(vocab_path=VOCAB_PATH, decoder_path=DECODER_PATH,
                   embed_type='numpy'):
    """Load saved trained embeddings and corresponding vocabulary

    Args:
        vocab_path (str): path to vocabulary pkl file
        decoder_path (str): path to trained decoder weights pkl file
        embed_type (str): whether to return embedding matrix as 'numpy' array or
                         as torch 'tensor'
    """
     # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    # Load the trained model parameters
    embeddings = torch.load(decoder_path)['embed.weight']

    if embed_type == 'numpy':
        embeddings = embeddings.detach().numpy()

    return embeddings, vocab


def main(args):
    embeddings, vocab = get_embeddings(vocab_path=args.vocab_path,
                                       decoder_path=args.decoder_path,
                                       embed_type=args.embed_type)
    # Check embedding shape
    print(embeddings.shape)
    # Check vocabulary
    print(len(vocab))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder_path', type=str, default=DECODER_PATH, help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default=VOCAB_PATH, help='path for vocabulary wrapper')
    parser.add_argument('--embed_type', type=str, default='numpy', help='type of embedding matrix')
    args = parser.parse_args()
    main(args)
