import sys
import os
import argparse
sys.path.append('../')
import config as conf
sys.path.append(os.path.join(conf.core_models_path))
sys.path.append(conf.utils_path)
from vocabulary import Vocabulary
from image_captioning.embedding_utils import get_embeddings as get_caption_embeddings
from glove.embedding_utils import get_embeddings as get_glove_embeddings
import word_similarity

def evaluate_similarity(embeddings, vocab):
	word_similarity.full_word_similarity_evaluation(embeddings, vocab, verbose=True)

def view_neighbors(embeddings, vocab, anchor, num_show=5):
	neighbors, dists = word_similarity.neighbors(anchor, embeddings, vocab)
	for i in neighbors[:num_show]:
		print('{}\t{}'.format(vocab.idx2word[i], dists[i]))

def main(args):
	embed_type = args.embed
	if embed_type == "caption":
		embeddings, vocab = get_caption_embeddings()
	elif embed_type == "glove":
		embeddings, vocab = get_glove_embeddings()
	elif embed_type == "bimodal":
		pass
	elif embed_type == "color":
		pass
	if args.similarity:
		evaluate_similarity(embeddings, vocab)
	if args.neighbors:
		anchor = args.neighbors
		view_neighbors(embeddings, vocab, anchor, num_show=6)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--embed', type=str, default='glove', help='Embeddings to be evaluated [glove, bimodal, color, caption]')
	parser.add_argument('--similarity', type=bool, default=False, help='Whether to evaluate similarity')
	parser.add_argument('--neighbors', type=str, default='glove', help='Word to find neighbors of')
	args = parser.parse_args()
	main(args)
