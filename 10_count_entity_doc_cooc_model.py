import argparse
import collections
import multiprocessing
import logging
import os 
import re

from tqdm import tqdm

def count_doc_coocs(path_vocab):

    path = path_dict[0]
    ent_vocab = path_dict[1]

    formula_one = '(?<=\[\[)(.+?)\|.+?(?=\]\])'
    formula_two = '(?<=\[\[)(.+?)(?=\]\])'
    ents = set()
    for line in open(os.path.join(root, fname)):
        #line = re.sub('_', ' ', line)
        line = re.sub(formula_one, r'\1', line)
        matches = re.findall(formula_two, line)
        #replacements = [re.sub('[^A-Za-z0-9]', '_', match) for match in matches]
        #for original, repl in zip(matches, replacements):
        #    line = line.replace('[[{}]]'.format(original), repl)
        #line = re.sub('[^A-Za-z0-9_ ]', ' ', line).lower().split()
        #if len(line) > 10:
        #    yield line
        for match in matches:
            try:
                current_ent = ent_vocab[e]
            except KeyError:
                continue
            ents.update(current_ent)
    return ents

parser = argparse.ArgumentParser()
parser.add_argument(
                    '--language', 
                    choices=['it', 'en'], 
                    required=True
                    )
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
if args.language == 'it':
    folder = '/import/cogsci/andrea/dataset/corpora/wexea_it/articles_2/'
elif args.language == 'en':
    folder = '/import/cogsci/andrea/dataset/corpora/wexea_annotated_wiki/ready_corpus/original_articles/'

formula_one = '(?<=\[\[)(.+?)\|.+?(?=\]\])'
formula_two = '(?<=\[\[)(.+?)(?=\]\])'
### counter
ent_dict = dict()
with tqdm() as counter:
    for root, _, files in os.walk(folder):
        for f in files:
            for line in open(os.path.join(root, f)):
                line = re.sub(formula_one, r'\1', line)
                matches = re.findall(formula_two, line)
                for match in matches:
                    try:
                        ent_dict[match] += 1
                    except KeyError:
                        ent_dict[match] = 1
                    counter.update(1)
### creating vocabulary        
ent_vocab = {k : k_i for k_i, k in enumerate(ent_dict.items()) if k[1]>100}

with multiprocessing.Pool(processes=os.cpu_count()) as pool:
    results = pool.map(count_doc_coocs, [[os.path.join(root, f), ent_vocab] for root, _, files in os.walk(folder) for f in files])
    pool.terminate()
    pool.join()

for exp_id in ['one', 'two']:
    ### joining all sets
    ### reading entity list
    entity_list_file = os.path.join('brain_data', exp_id, 'wikidata_ids_{}.txt'.format(exp_id))
    assert os.path.exists(entity_list_file)
    with open(entity_list_file) as i:
        ent_lines = [l.strip().split('\t') for l in i.readlines()]
    for l_i, l in enumerate(ent_lines):
        if l_i == 0:
            print(l)
            assert l[0] == 'entity'
            assert l[1] == 'wikidata_id'
        assert len(l) == 2
    entities = {l[0] : l[1] for l in ent_lines[1:] if l[0][0]!='#'}
    vecz = {k : numpy.zeros(len(ent_vocab.keys())) for k in entities.keys()}
    for res in results:
        for k in vecz.keys():
            if k in res:
                for cooc_e in [e for e in res if e!=e]:
                    vekz[k][cooc_e] += 1
