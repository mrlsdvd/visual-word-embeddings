import sys, os
import csv

sys.path.append('../')
import config as conf
wordsim_home = os.path.join(conf.processed_data_path, 'evaluation', 'wordsim')

def wordsim_dataset_reader(src_filename, header=False, delimiter=','):
    """Basic reader that works for all four files, since they all have the
    format word1,word2,score, differing only in whether or not they include
    a header line and what delimiter they use.

    Parameters
    ----------
    src_filename : str
        Full path to the source file.
    header : bool (default: False)
        Whether `src_filename` has a header.
    delimiter : str (default: ',')
        Field delimiter in `src_filename`.

    Yields
    ------
    (str, str, float)
       (w1, w2, score) where `score` is the negative of the similarity
       score in the file so that we are intuitively aligned with our
       distance-based code.

    """
    with open(src_filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if header:
            next(reader)
        for row in reader:
            w1, w2, score = row
            # Negative of scores to align intuitively with distance functions:
            score = -float(score)
            yield (w1, w2, score)

def wordsim353_reader():
    """WordSim-353: http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/"""
    src_filename = os.path.join(wordsim_home, 'wordsim353.csv')
    return wordsim_dataset_reader(src_filename, header=True)

def mturk287_reader():
    """MTurk-287: http://tx.technion.ac.il/~kirar/Datasets.html"""
    src_filename = os.path.join(wordsim_home, 'MTurk-287.csv')
    return wordsim_dataset_reader(src_filename, header=False)

def mturk771_reader():
    """MTURK-771: http://www2.mta.ac.il/~gideon/mturk771.html"""
    src_filename = os.path.join(wordsim_home, 'MTURK-771.csv')
    return wordsim_dataset_reader(src_filename, header=False)

def men_reader():
    """MEN: http://clic.cimec.unitn.it/~elia.bruni/MEN"""
    src_filename = os.path.join(wordsim_home, 'MEN_dataset_natural_form_full')
    return wordsim_dataset_reader(src_filename, header=False, delimiter=' ')

def rg65_reader():
    """RG-65: https://www.dropbox.com/s/chopke5zqly228d/EN-RG-65.txt"""
    src_filename = os.path.join(wordsim_home, 'EN-RG-65.txt')
    return wordsim_dataset_reader(src_filename, header=False, delimiter='\t')
