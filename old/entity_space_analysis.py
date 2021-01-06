import collections

import numpy

from utils.tsne_plots import tsne_plot_words
from utils.general_utilities import load_word_vectors

models = ['wiki2vec', 'word2vec', 'bert']

for m in models:
    words, vector_dimensions = load_word_vectors(m, categories=True)

    keys = [k for k in words.keys()]

    dict_one = collections.defaultdict(numpy.ndarray)
    dict_two = collections.defaultdict(numpy.ndarray)

    midway = int(len(keys)/2)

    for i in range(midway):
        dict_one[keys[i]] = words[keys[i]]
        dict_two[keys[i+midway]] = words[keys[i+midway]]

    title = 'TSNE plot comparing place and person vectors in {}'.format(m)
    filename = 'temp/tsne_categories_{}.png'.format(m)

    tsne_plot_words(title, dict_one, dict_two, filename)
