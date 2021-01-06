import os
import re
import gensim
import logging
import string

from gensim import utils

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __init__(self):
        self.punct = string.punctuation

    def __iter__(self):
        corpus_path = '/import/cogsci/andrea/dataset/wexea_annotated_wiki/clean_articles'
        for root, directories, files in os.walk(corpus_path):
            for f in files: 
                collection = []
                for line in open(os.path.join(root, f)):
                    if len(collection) < 256:
                        line = [re.sub('[{}\d]'.format(self.punct), '', w).lower() if '[[[' not in w else w for w in line.lstrip().rstrip().split(' ')]
                        line = [w for w in line if w != '' and len(w) > 1]
                        collection += line
                    else:
                        yield collection 
                        collection = []

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = MyCorpus()
model = gensim.models.Word2Vec(sentences=sentences, size=300, workers=48, max_vocab_size=500000)
model.save('models/w2v_entities')
