import argparse
import logging
import os 
import re

from gensim.models import Word2Vec

class MySentences(object):
        def __init__(self, dirname):
            self.dirname = dirname

        def __iter__(self):

            formula_one = '(?<=\[\[)(.+?)\|.+?(?=\]\])'
            formula_two = '(?<=\[\[)(.+?)(?=\]\])'

            for root, direc, filez in os.walk(self.dirname):
                for fname in filez:
                    for line in open(os.path.join(root, fname)):
                        line = re.sub('_', ' ', line)
                        line = re.sub(formula_one, r'\1', line)
                        matches = re.findall(formula_two, line)
                        replacements = [re.sub('[^A-Za-z0-9]', '_', match) for match in matches]
                        for original, repl in zip(matches, replacements):
                            line = line.replace('[[{}]]'.format(original), repl)
                        line = re.sub('[^A-Za-z0-9_ ]', ' ', line).lower().split()
                        if len(line) > 10:
                            yield line

parser = argparse.ArgumentParser()
parser.add_argument(
                    '--language', 
                    choices=['it', 'en'], 
                    required=True
                    )
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
if args.language == 'it':
    min_count = 20
    folder = '/import/cogsci/andrea/dataset/corpora/wexea_it/articles_2/'
elif args.language == 'en':
    min_count = 100
    folder = '/import/cogsci/andrea/dataset/corpora/wexea_annotated_wiki/ready_corpus/original_articles/'
model = Word2Vec(
                 sentences=MySentences(folder), 
                 size=300, 
                 window=10, 
                 min_count=min_count, 
                 negative=10,
                 sample=1e-5,
                 sg=0,
                 workers=int(os.cpu_count()/4)
                 )
model.save("word2vec_wexea_entities_en_win_10_min_{}_neg_1e-5_size_300.model".format(min_count))
