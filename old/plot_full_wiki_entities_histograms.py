import matplotlib
import collections

from matplotlib import pyplot
from utils.extract_word_lists import Entities
from tqdm import tqdm

ents, categories = Entities('full_wiki').words

to_be_plotted = collections.defaultdict(int)

for coarse, finer in categories.items():
    to_be_plotted[coarse] = [v[-1] for k, v in ents.items() if v[0] == coarse]
    for fine in finer.keys():
        to_be_plotted[fine] = [v[-1] for k, v in ents.items() if v[1] == fine]

for cat, lst in tqdm(to_be_plotted.items()):
    pyplot.hist(lst, bins=20)
    pyplot.title='Histogram for {} entities - N={}'.format(cat, len(lst))
    pyplot.savefig('temp/{}_histogram.png'.format(cat))
    pyplot.clf()

for cat, lst in tqdm(to_be_plotted.items()):
    pyplot.hist(lst, log=True, bins=20)
    pyplot.savefig('temp/log_{}_histogram.png'.format(cat))
    pyplot.clf()
