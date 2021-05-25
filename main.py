import argparse
import os
import re

from extract_word_lists import Entities
from clean_sentences import read_sentences
from extract_bert import bert
from extract_elmo import elmo
from extract_transe import transe
from tqdm import tqdm

def vector_to_txt(word, vector_layers, output_file):
    output_file.write('{}\n'.format(word))
    for vector in vector_layers:
        for dimension_value in vector:
            output_file.write('{}\t'.format(dimension_value))
        output_file.write('\n')

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str, \
                    choices=['bert', 'elmo', 'ernie', \
                            'transe'], \
                    help='Indicates which \
                    model to extract the vectors for')
parser.add_argument('--extraction_mode', default='full_sentence', type=str, \
                    choices=['unmasked', 'masked', 'full_sentence'], \
                    help='Indicates which masking mode to use')
parser.add_argument('--output_folder', required=True, type=str, \
                    help='Where to store the extracted word vectors')
parser.add_argument('--word_selection', required=True, type=str, \
                    choices = ['full_wiki', 'men', 'stopwords', \
                               'eeg_one', 'eeg_stanford', 'wakeman_henson'], \
                    help='The words whose vectors will be extracted')
args = parser.parse_args()


if args.model in ['elmo', 'bert', 'ernie']:
    out_folder = os.path.join(args.output_folder, args.model, args.extraction_mode)
else:
    out_folder = os.path.join(args.output_folder, args.model)
os.makedirs(out_folder, exist_ok=True)

ent_dict = Entities(args.word_selection).word_categories

if args.word_selection == 'wikisrs':
    b = Entities('full_wiki').words
    w_list = list()
    for w in ent_dict.keys():
        if w not in b.keys():
            w_list.append(w)

    ent_dict = {k : 'ent' for k in w_list}

if args.model != 'transe':
    if args.model == 'bert':
        ent_dict = {k : v for k, v in ent_dict.items() if k != 'Spain' and k != 'Germany'}
    ent_sentences = read_sentences(ent_dict, args)

    if args.model == 'elmo':
        entity_vectors = elmo(ent_sentences, args)
    elif args.model == 'bert':
        entity_vectors = bert(ent_sentences, args)
    elif args.model == 'ernie':
        from extract_ernie import ernie
        entity_vectors = ernie(ent_sentences, args)

elif args.model == 'transe':
    entity_vectors = transe(ent_dict, args)

print('Now writing vectors to file...')
for entity, vector_tuples in tqdm(entity_vectors.items()):

    with open(os.path.join(out_folder, '{}.vec'.format(re.sub(' ', '_', entity))), 'w') as o:
        try:
            #for sentence, vector_layers in [vector_tuples]:
            for sentence, vector_layers in vector_tuples:
                
                vector_to_txt(sentence, vector_layers, o)
        except ValueError:
            print(entity)
