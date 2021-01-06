# It should be moved to the data folder, where the ERNIE folder is situated

import torch
import numpy
import pickle
import logging
import gensim
import re
import random
import os
import argparse
import collections

from wikipedia2vec import Wikipedia2Vec
from tqdm import tqdm
from collections import defaultdict
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.extract_word_lists import Entities

'''
def get_identities(general_info=False):

    with open('resources/wiki_stimuli.txt') as ids_file:
        wiki2vec_ids = [((re.sub('.bmp', '', l)).strip('\n')).split('\t') for l in ids_file.readlines()]
        if general_info:
            general_info = {l[0] : l[2:] for l in wiki2vec_ids if len(l) > 2}
        fmri2wiki = {l[0] : l[1] for l in wiki2vec_ids if len(l) > 2}
    with open('resources/transe_stimuli.txt') as ids_file:
        transe_ids = [((re.sub('.bmp', '', l)).strip('\n')).split('\t') for l in ids_file.readlines()]
        fmri2transe = {l[0] : l[1] for l in transe_ids if len(l) > 2}
    assert len(wiki2vec_ids) == len(transe_ids)
    assert [k for k in fmri2wiki.keys()] == [k for k in fmri2transe.keys()]

    if general_info:
        return fmri2wiki, fmri2transe, general_info
    else:
        return fmri2wiki, fmri2transe 

def get_men_words():
    words = []
    with open('resources/men.txt', 'r') as men_file:
        all_lines = [l.split()[:2] for l in men_file.readlines()]
    all_words = [i for i in {re.sub('_.', '', w) : '' for l in all_lines for w in l}.keys()]
    return all_words

def get_uk_words():
    files = ['people', 'places']
    entity_words = []
    generic_words = []
    wikidata_identifiers = []
    for f in files:
        with open('resources/uk_{}.txt'.format(f), 'r') as uk_file:
            all_lines = [l.strip().split('\t') for l in uk_file.readlines()]
        for l in all_lines:
            entity_words.append(l[0])
            generic_words.append(l[1])
            wikidata_identifiers.append(l[2])
    return entity_words, generic_words, wikidata_identifiers

def get_wikidata_words():
    with open('models/TransE/entity_map.txt') as mapping_file:
        id_map = collections.defaultdict(str)
        for l in mapping_file.readlines():
            line = l.strip().split('\t')
            id_map[line[0]] = line[1]

    people_and_places = collections.defaultdict(dict)

    with open('resources/entities_counter.txt', 'r') as entities_file:
        all_lines = [l.strip().split('\t') for l in entities_file.readlines() if '\tPlace' in l or '\tPerson' in l]
    for l in all_lines:
        #person_words.append(l[0]) if l[1] == 'Person' else places_words.append(l[0])
        name = re.sub('_', ' ', l[0])
        try:
            unified_id = id_map[name]
            if l[2] == 'Person':
                try:
                    people_and_places['people'][name] = (unified_id, l[3])
                except IndexError:
                    people_and_places['people'][name] = (unified_id, 'unknown')
            else:
                try:
                    people_and_places['places'][name] = (unified_id, l[3])
                except IndexError:
                    people_and_places['places'][name] = (unified_id, 'unknown')
        except KeyError:
            print('Couldn\'t find {}'.format(name))
           
    return people_and_places

def get_stopwords():
    words = []
    with open('resources/stopwords.txt', 'r') as stopwords_file:
        all_words = [l.strip('\n' ) for l in stopwords_file.readlines()[1:] if len(l) >= 5]
    return all_words
'''

def vector_to_txt(word, vector, output_file):
    output_file.write('{}\t'.format(word))
    for dimension_index, dimension_value in enumerate(vector):
        if dimension_index != len(vector)-1:
            output_file.write('{}\t'.format(dimension_value))
        else: 
            output_file.write('{}\n'.format(dimension_value))

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, choices=['men', 'individuals', 'uk', 'wikidata'], help='Specify where to pick up the word list from')
parser.add_argument('--bert_layer', type=int, default=11, help='Indicates which layer to use for extracting the word vectors from BERT')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

os.makedirs('word_vectors', exist_ok=True)

fmri2wiki, fmri2transe = get_identities()

if args.data_name == 'men':

    men_words_ordered = get_men_words()
    random_indices = random.sample([k for k in range(len(men_words_ordered))], len(fmri2wiki))
    words = [men_words_ordered[i] for i in random_indices]

elif args.data_name == 'uk':
    entity_words, words, wikidata_identifiers = get_uk_words()

elif args.data_name == 'wikidata':
    people_and_places = get_wikidata_words()

stopwords = get_stopwords()[:len(fmri2wiki)]
ids = [k for k in fmri2wiki.keys()]

wiki2vec = Wikipedia2Vec.load('models/wikipedia2vec/models/enwiki_20180420_win10_100d.pkl')
#word2vec = gensim.models.Word2Vec.load('models/w2v_background_space/wiki_w2v_2018_size300_window5_max_final_vocab250000_sg1')
word2vec = gensim.models.Word2Vec.load('models/w2v_entities')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
'''
# Wikipedia2Vec
print('Wiki2Vec')

for data_category, proper_names in people_and_places.items():

    with open('word_vectors/wiki2vec_{}_{}.vec'.format(args.data_name, data_category), 'w') as o:

        if args.data_name == 'uk' or args.data_name == 'wikidata':

            for proper_name in proper_names.keys():

                try:

                    #entity_vector = wiki2vec.get_entity_vector(entity_words[word_index])
                    entity_vector = wiki2vec.get_entity_vector(proper_name)
                    #vector_to_txt(entity_words[word_index], entity_vector, o)
                    vector_to_txt(proper_name, entity_vector, o)

                except KeyError:
                    #print('Could not retrieve word vector: {}'.format(entity_words[word_index]))
                    print('Could not retrieve word vector: {}'.format(proper_name))

        elif args.data_name != 'individuals':

            for word_index, word in enumerate(words):
                try:
                    vector = wiki2vec.get_word_vector(word.lower())
                    vector_to_txt(word, vector, o)

                except KeyError:
                    print('Could not retrieve word vector: {}'.format(word))



        else:
            for identity in ids:

                wiki_name = fmri2wiki[identity]

                try:
                    entity_vector = wiki2vec.get_entity_vector(wiki_name)
                    vector_to_txt(identity, entity_vector, o)
                except KeyError:
                    print('Could not retrieve entity: {}'.format(word))

# Word2Vec

print('W2V')
for data_category, proper_names in people_and_places.items():

    with open('word_vectors/word2vec_{}_{}.vec'.format(args.data_name, data_category), 'w') as o:

        if args.data_name == 'wikidata':

            for proper_name in proper_names.keys():
                
                proper_name_query = '[[[{}]]]'.format(re.sub(' ', '_', proper_name))

                try:
                    vector = word2vec.wv[proper_name_query]
                    vector_to_txt(proper_name, vector, o)
                except KeyError:
                    print('Could not retrieve word vector: {}'.format(proper_name))

        elif args.data_name != 'individuals':

            for word_index, word in enumerate(words):

                try:
                    vector = word2vec.wv[word.lower()]
                    vector_to_txt(word, vector, o)
                except KeyError:
                    print('Could not retrieve word vector: {}'.format(word))

                if args.data_name == 'uk':

                    entity_word = [w.lower() for w in entity_words[word_index].split() if len(w)>2 and w!='River' and w!='Cathedral']
                    if len(entity_word) > 1:
                        entity_word = entity_word[1]
                    else:
                        entity_word = entity_word[0]
                    final_vector = numpy.zeros(300)

                    for word in entity_word:
                        try:
                            entity_vector = word2vec.wv[word]
                            final_vector = numpy.add(final_vector, entity_vector)

                        except KeyError:
                            print('Could not retrieve word vector: {}'.format(word))

                    vector_to_txt(entity_word, final_vector, o)

        else:
            for identity in ids:

                transe_name = fmri2transe[identity]
                words = [re.sub('\W', '', w).lower() for w in transe_name.split() if w != 'of']

                final_vector = numpy.zeros(300)
                for word in words:
                    try:
                        vector = word2vec.wv[word]
                        final_vector = numpy.add(final_vector, vector)
                    except KeyError:
                        print('Could not retrieve word vector: {}'.format(word))
                        vector = numpy.zeros(300)

                vector_to_txt(identity, final_vector, o)

# Create an individual vector for Bert 
print('BERT individual')

for data_category, proper_names in people_and_places.items():

    with open('word_vectors/bert_{}_{}.vec'.format(args.data_name, data_category), 'w') as o:
        if args.data_name != 'individuals':
            #for word_index, word in enumerate(words):
            for proper_name in proper_names.keys():
                short_folder = proper_name[:3].lower()
                txt_file = '{}.txt'.format(re.sub(' ', '_', proper_name))
                try:
                    with open(os.path.join('/import/cogsci/andrea/dataset/wexea_annotated_wiki/ready_corpus/original_articles', short_folder, txt_file)) as bert_txt:
                        bert_input = []
                        counter = 0
                        for l in bert_txt:
                            if len(bert_input) < 64:
                                line = [w.split('[[') for w in l.strip().split(']]')]
                                bert_line = []
                                for words in line:
                                    for w in words:
                                        if '{}|'.format(proper_name) in w and counter == 0:
                                            bert_line.append('[MASK]')
                                            counter += 1
                                        else:
                                            bert_line.append(re.sub('^.+\|', '', w))
                                bert_line = ''.join(bert_line).split()
                                bert_input += bert_line
                            
                            else:
                                if '[MASK]' not in bert_input:
                                    bert_input = ['[MASK]'] + bert_input
                                if len(bert_input) > 400:
                                    bert_input = bert_input[:400]
                                bert_input = ' '.join(bert_input)
                                #print(bert_input)
                                break 
                        
                    
                    #print(bert_tokenizer.tokenize(proper_name))
                    #input_ids = torch.tensor([bert_tokenizer.encode(word)])
                    tokenized_input = bert_tokenizer.encode(bert_input)
                    input_ids = torch.tensor([tokenized_input])
                    masked_index = [i for i, v in enumerate(tokenized_input) if v == 103][0]
                    with torch.no_grad():
                        output = bert_model(input_ids)
                        final_vector = numpy.zeros(768)
                        #for i in range(input_ids.shape[1]-2):
                            #index = i+1
                            #final_vector = numpy.add(final_vector, output[2][5][0][index].numpy())
                        final_vector = output[2][12][0][masked_index].numpy()

                    #vector_to_txt(word, final_vector, o)
                    vector_to_txt(proper_name, final_vector, o)
                except FileNotFoundError:
                    print('impossible to extract the word vector for {}'.format(proper_name))


                if args.data_name == 'uk':

                    entity_word = [w for w in entity_words[word_index].split() if len(w)>2 and w!='River' and w!='Cathedral']
                    if len(entity_word) > 1:
                        entity_word = entity_word[1]
                    else:
                        entity_word = entity_word[0]
                    input_ids = torch.tensor([bert_tokenizer.encode(entity_word)])
                    with torch.no_grad():
                        output = bert_model(input_ids)
                        final_vector = numpy.zeros(768)
                        for i in range(input_ids.shape[1]-2):
                            index = i+1
                            final_vector = numpy.add(final_vector, output[2][1][0][index].numpy())

                    vector_to_txt(entity_word, final_vector, o)

        else:
            for identity in ids:

                transe_name = fmri2transe[identity]
                input_ids = torch.tensor([bert_tokenizer.encode(transe_name)])
                with torch.no_grad():
                    output = bert_model(input_ids)
                    final_vector = numpy.zeros(768)
                    for i in range(input_ids.shape[1]-2):
                        index = i+1
                        final_vector = numpy.add(final_vector, output[2][1][0][index].numpy())

                vector_to_txt(identity, final_vector, o)
'''
'''
if args.data_name != 'men':

    # Transe

    with open('word_vectors/transe_individuals_two.vec', 'w') as o:

        experiment_names_and_ids = {v : k for k, v in fmri2transe.items()}

        with open('models/TransE/entity_map.txt', 'r') as input_file:
            people_transe = [k.strip('\n').split('\t') for k in input_file.readlines()]

        reduced_transe_dict = {k[0] : k[1] for k in people_transe if k[0] in experiment_names_and_ids.keys()}
        full_transe_dict = {k[0] : k[1] for k in people_transe}
        from_Q_to_names = {v : k for k, v in reduced_transe_dict.items()}
        reduced_transe_Q_ids = [v for k, v in reduced_transe_dict.items()]

        with open('models/TransE/entity2id.txt', 'r') as input_file:
            lines = [l.strip('\n').split('\t') for l in input_file.readlines()]
            experiment_ids = {l[0] : int(l[1]) for l in lines if l[0] in reduced_transe_Q_ids}
            assert min([int(k[1]) for k in lines]) == 0
            max_index = max([int(k[1]) for k in lines])

        indices_to_names = {v : from_Q_to_names[k] for k, v in experiment_ids.items()} 
        names_to_indices = {v : k for k, v in indices_to_names.items()} 
        transe_vectors = defaultdict(numpy.array)
        with open("models/TransE/entity2vec.vec", 'r') as vector_file:    
            lines = [k for k in vector_file.readlines()]
            assert max_index == len(lines) - 1
            for line_index, line in enumerate(lines):
                if line_index in indices_to_names.keys():
                    line_name = indices_to_names[line_index]
                    transe_vectors[experiment_names_and_ids[line_name]] = line.strip().split('\t')
        for identity in ids:
            o.write('{}\t'.format(identity))
            line = transe_vectors[identity]
            for word_index, w in enumerate(line):
                if word_index != len(line)-1:
                    o.write('{}\t'.format(w))
                else:
                    o.write('{}\n'.format(w))

    # Ernie
    from models.ERNIE.knowledge_bert import BertTokenizer as ErnieTokenizer
    from models.ERNIE.knowledge_bert import BertModel as ErnieModel
    ernie_tokenizer = ErnieTokenizer.from_pretrained('models/ERNIE/ernie_base')
    ernie_model, _ = ErnieModel.from_pretrained('models/ERNIE/ernie_base')
    ernie_model.eval()

    transe_to_torch = torch.nn.Embedding.from_pretrained(torch.FloatTensor([[0]*100] + [numpy.asarray(v, dtype=numpy.single) for k, v in transe_vectors.items()]))
    name_to_reduced_indices = {fmri2transe[k] : i for i, k in enumerate([n for n in transe_vectors.keys()])}
    # Tokenize
    sentences = {k : ernie_tokenizer.tokenize(k, [[v, 0, 0, 0]]) for k, v in reduced_transe_dict.items()}
    #old_vectors = pickle.load(open('pickles/ernie_vecs.pkl', 'rb'))
    ernie_vectors = defaultdict(list)

    for individual, tokens_and_entities in sentences.items():
        tokens = ["[CLS]"] + tokens_and_entities[0] + ["[SEP]"] 
        entity_tokens = ["UNK"] + tokens_and_entities[1] + ["UNK"]
        input_mask = [1] * len(tokens)

        # Convert token to vocabulary indices
        indexed_tokens = ernie_tokenizer.convert_tokens_to_ids(tokens)
        sentence = [entity_tokens, indexed_tokens]
        indexed_ents = []
        ent_mask = []
        for ent_index, ent in enumerate(entity_tokens):
            if ent != "UNK":
                indexed_ents.append(name_to_reduced_indices[individual])
                relevant_index = ent_index
                ent_mask.append(1)
            else:
                indexed_ents.append(-1)
                ent_mask.append(0)
        ent_mask[0] = 1

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        ents_tensor = torch.tensor([indexed_ents])
        ent_mask = torch.tensor([ent_mask])

        # If you have a GPU, put everything on cuda
        #tokens_tensor = tokens_tensor.to('cuda')
        ents_tensor = transe_to_torch(ents_tensor+1)
        #ent_mask = ent_mask.to("cuda")
        #ernie_model.to('cuda')

        # Predict all tokens
        with torch.no_grad():
            encoded_layers = ernie_model(input_ids=tokens_tensor, input_ent=ents_tensor, ent_mask=ent_mask)
            #ernie_vectors[experiment_names_and_ids[individual]] = encoded_layers[0][11][0][relevant_index].cpu().numpy()
            ernie_vectors[experiment_names_and_ids[individual]] = encoded_layers[0][0][0][relevant_index].cpu().numpy()

    with open('word_vectors/ernie_two_individuals.vec', 'w') as o:
        for identity in ids:
            o.write('{}\t'.format(identity))
            vector = ernie_vectors[identity]
            for dimension, dimension_value  in enumerate(vector):
                if dimension != len(vector)-1:
                    o.write('{}\t'.format(dimension_value))
                else:
                    o.write('{}\n'.format(dimension_value))
'''
