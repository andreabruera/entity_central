import argparse
import numpy
import os
import re
import spacy

from gensim.models import Word2Vec
from qwikidata.linked_data_interface import get_entity_dict_from_api
from tqdm import tqdm
from wikipedia2vec import Wikipedia2Vec

from utils import load_vec_files, read_entity_sentences
from original_ratings_utils import read_datasets, read_personally_familiar_sentences

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_id', choices=['one', 'two'], required=True)
parser.add_argument(
                    '--language', 
                    choices=['it', 'en'], 
                    required=True
                    )
parser.add_argument('--model', 
                    choices=[
                             'imageability',
                             'familiarity',
                             'concreteness',
                             'valence',
                             'arousal',
                             'dominance',
                             'perceptual',
                             'affective',
                                ],
                    required=True, help='Which ratings?')
args = parser.parse_args()

missing = list()
spacy_model = spacy.load("it_core_news_lg")
datasets = read_datasets()
spacyed_datasets = {k : dict() for k in datasets.keys()}
for k, v in datasets.items():
    for stim, rating in v.items():
        stim = [tok.lemma_ for tok in spacy_model(stim)][0]
        spacyed_datasets[k][stim] = rating 

args.corpus = 'wikipedia'
args.corpus_portion = 'entity_sentences'
args.layer = 'top_four'
args.individuals = True
args.random = False
vecs_file, rankings_folder = load_vec_files(args, args.model)

entity_sentences = read_entity_sentences(args)
if args.experiment_id == 'two':
    pers_sentences = read_personally_familiar_sentences()
    for sub, sub_dict in pers_sentences.items():
        for name, sents in sub_dict.items():
            entity_sentences['{:02}_{}'.format(sub, name)] = sents
relevant_pos = [
           'ADJ', 
           'ADV', 
           'NOUN', 
           'PROPN', 
           'VERB',
           #'X',
           'NUM',
           'AUX',
           ]
to_be_ignored = ['OTHER_NAME', 'STREET_NAME', 'PERSON_NAME', 'PROPER_NAME', 'PEOPLE_NAME']

vectors = dict()
with tqdm() as counter:
    for name, sentences in entity_sentences.items():
        missing = list()
        used = list()

        ent_vecs = list()
        formula_one = '(?<=\[\[)(.+?)\|.+?(?=\]\])'
        formula_two = '(?<=\[\[)(.+?)(?=\]\])'

        for line in sentences:
            line = re.sub('_', ' ', line)
            line = re.sub(formula_one, r'\1', line)
            matches = re.findall(formula_two, line)
            replacements = [re.sub('[^A-Za-z0-9]', '_', match) for match in matches]
            for original, repl in zip(matches, replacements):
                line = line.replace('[[{}]]'.format(original), repl)
            #line = re.sub('[^A-Za-z0-9_ ]', ' ', line).lower().split()
            line = re.sub('[^A-Za-z0-9_ ]', ' ', line)
            line = line.lower()
            if len(line.split()) > 10:
                spacy_line = spacy_model(line)
                line_vec = list()
                #print(line)
                ### entities
                ### words
                for tok in spacy_line:
                    #if w not in stopwords:
                    if len(tok.text) < 3:
                        #print(tok.text)
                        continue
                    if (tok.pos_ not in relevant_pos and '_' not in tok.text) or tok.text in to_be_ignored:
                        print((tok.pos_, tok.text))
                        continue
                    try:
                        line_vec.append(spacyed_datasets[args.model][tok.lemma_])
                        used.append(tok.text)
                    except KeyError:
                        missing.append(tok.text)
                try:
                    assert len(line_vec) >= 1
                except AssertionError:
                    print(line)
                    continue
                line_vec = numpy.average(line_vec, axis=0)
                ent_vecs.append((line, line_vec))
                counter.update(1)
        assert len(ent_vecs) >= 1
        vectors[name] = ent_vecs

#for k, v in vectors.items():
    #print(k)
    #for line, vec in v:
    #    assert vec.shape == (desired_shape,)
### writing to file the vectors

with open(vecs_file, 'w') as o:
    for stim, vecz in vectors.items():
        for line, dim in vecz:
            line = line.replace('\t', ' ')
            o.write('{}\t{}\t'.format(stim, line))
            if args.model in ['perceptual', 'affective']:
                for vec in dim:
                    o.write('{}\t'.format(float(vec)))
            else:
                #for dim in vec:
                o.write('{}\t'.format(float(dim)))
                #o.write('\n')
            o.write('\n')
