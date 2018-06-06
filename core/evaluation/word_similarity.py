import sys
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
sys.path.append('../')
import config as conf
sys.path.append(conf.utils_path)
import vsm
from wordsim_dataset_reader import wordsim353_reader, men_reader, mturk287_reader, mturk771_reader, rg65_reader

def word_similarity_evaluation(reader, embeddings, vocabulary, distfunc=vsm.cosine, verbose=True):
    """Word-similarity evalution framework.

    Parameters
    ----------
    reader : iterator
        A reader for a word-similarity dataset. Just has to yield
        tuples (word1, word2, score).
    embeddings : np.array
        The matrix of word embeddings
    vocabulary : Vocabulary
        Maps word to corresponding row index in embedding matrix
    distfunc : function mapping vector pairs to floats (default: `vsm.cosine`)
        The measure of distance between vectors. Can also be `vsm.euclidean`,
        `vsm.matching`, `vsm.jaccard`, as well as any other distance measure
        between 1d vectors.
    verbose : bool
        Whether to print information about how much of the vocab
        `df` covers.

    Prints
    ------
    To standard output
        Size of the vocabulary overlap between the evaluation set and
        rownames. We limit the evalation to the overlap, paying no price
        for missing words (which is not fair, but it's reasonable given
        that we're working with very small VSMs in this notebook).

    Returns
    -------
    float
        The Spearman rank correlation coefficient between the dataset
        scores and the similarity values obtained from `mat` using
        `distfunc`. This evaluation is sensitive only to rankings, not
        to absolute values.

    """
    sims = defaultdict(list)
    rownames = vocabulary.word2idx.keys()
    eval_vocab = set()
    excluded = set()
    for w1, w2, score in reader():
        if w1 in rownames and w2 in rownames:
            sims[w1].append((w2, score))
            sims[w2].append((w1, score))
            eval_vocab |= {w1, w2}
        else:
            excluded |= {w1, w2}
    all_words = eval_vocab | excluded
    if verbose:
        print("Evaluation vocab: {:,} of {:,}".format(len(eval_vocab), len(all_words)))
    # Evaluate the matrix by creating a vector of all_scores for data
    # and all_dists for mat's distances.
    all_scores = []
    all_dists = []
    for word in eval_vocab:
        # vec = df.loc[word]
        vec = embeddings[vocabulary(word)]
        vals = sims[word]
        cmps, scores = zip(*vals)
        all_scores += scores
        all_dists += [distfunc(vec, embeddings[vocabulary.word2idx[w]]) for w in cmps]
    rho, pvalue = spearmanr(all_scores, all_dists)
    return rho


def full_word_similarity_evaluation(embeddings, vocab, verbose=True):
    """Evaluate a VSM against all four datasets.

    Parameters
    ----------
    embeddings : np array of word embeddings
    vocab : dictionary mapping word to row index of corresponding embedding

    Returns
    -------
    dict
        Mapping dataset names to Spearman r values

    """
    scores = {}
    for reader in (wordsim353_reader, mturk287_reader, mturk771_reader, men_reader, rg65_reader):
        if verbose:
            print("="*40)
            print(reader.__name__)
        score = word_similarity_evaluation(reader, embeddings, vocab, verbose=verbose)
        scores[reader.__name__] = score
        if verbose:
            print('Spearman r: {0:0.03f}'.format(score))
    mu = np.array(list(scores.values())).mean()
    if verbose:
        print("="*40)
        print("Mean Spearman r: {0:0.03f}".format(mu))
    return scores


def neighbors(word, embeddings, vocabulary, distfunc=vsm.cosine):
    """Tool for finding the nearest neighbors of `word` in `df` according
    to `distfunc`. The comparisons are between row vectors.

    Parameters
    ----------
    word : str
        The anchor word. Assumed to be in vocabulary.
    embeddings : np.ndarray
        The vector-space model matrix.
    vocabulary : Vocabulary
        Vocabulary object containing word2idx and idx2word dicts
    distfunc : function mapping vector pairs to floats (default: `cosine`)
        The measure of distance between vectors. Can also be `euclidean`,
        `matching`, `jaccard`, as well as any other distance measure
        between 1d vectors.

    Raises
    ------
    ValueError
        If word is not in `df.index`.

    Returns
    -------
    pd.Series
        Ordered by closeness to `word`.

    """
    if word not in vocabulary.word2idx.keys():
        raise ValueError('{} is not in this VSM'.format(word))
    w_vec = embeddings[vocabulary(word)]
    dists = np.apply_along_axis(lambda x: distfunc(w_vec, x), axis=1, arr=embeddings)
    return dists.argsort(), dists
