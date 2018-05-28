import sys
import os


sys.path.append('../')
import config as conf

sys.path.append(os.path.join(conf.core_models_path, 'image_captioning'))

import embedding_utils
import word_similarity

def evaluate():
	embeddings, vocab = embedding_utils.get_embeddings()
	word_similarity.full_word_similarity_evaluation(embeddings, vocab, verbose=True)

evaluate()
