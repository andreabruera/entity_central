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
from exp_two_utils import read_personally_familiar_sentences

def Q_to_line(ernie_path):

    # Convert ents
    entity2id = {}
    with open(os.path.join(ernie_path, \
              "kg_embed/entity2id.txt")) as fin:
        fin.readline()
        for line in fin:
            qid, eid = line.strip().split('\t')
            entity2id[qid] = int(eid)

    return entity2id

def entity_to_Q_id(ernie_path):

    # Read entity map
    ent_map = {}
    with open(os.path.join(ernie_path, \
              "kg_embed/entity_map.txt")) as fin:
        for line in fin:
            name, qid = line.strip().split("\t")
            ent_map[name] = qid

    return ent_map

def get_transe_vectors(ernie_path):

    vecs = []
    #vecs.append([0]*100)
    with open(os.path.join(ernie_path, \
              "kg_embed/entity2vec.vec"), 'r') as fin:
        for line in tqdm(fin):
            vec = line.strip().split('\t')
            vec = [float(x) for x in vec]
            vecs.append(vec)

    return vecs

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_id', choices=['one', 'two'], required=True)
parser.add_argument(
                    '--language', 
                    choices=['it', 'en'], 
                    required=True
                    )
parser.add_argument('--model', 
                    choices=[
                             'wikipedia2vec',
                             'transe',
                             'w2v',
                             'w2v_sentence',
                             'wikipedia2vec_sentence',
                                ],
                    required=True, help='Which model?')
args = parser.parse_args()
out_folder = 'static_vectors'
os.makedirs(out_folder, exist_ok=True)

aliases = dict()
with open(os.path.join('brain_data', args.experiment_id, 'wikidata_ids_{}.txt'.format(args.experiment_id))) as i:
    ent_lines = [l.strip().split('\t') for l in i.readlines()]
for l_i, l in enumerate(ent_lines):
    if l_i == 0:
        print(l)
        assert l[0] == 'entity'
        assert l[1] == 'wikidata_id'
    assert len(l) == 2
entities = {l[0] : l[1] for l in ent_lines[1:] if l[0][0]!='#'}
for k, wikidata_id in tqdm(entities.items()):
    ent_dict = get_entity_dict_from_api(wikidata_id)
    main_alias = ent_dict['labels'][args.language]['value']
    aliases[k] = [k, main_alias]
    if args.language in ent_dict['aliases'].keys():
        for al in ent_dict['aliases'][args.language]:
            if len(al) > 2:
                aliases[k].append(al['value'])
    #if k == 'Spagna':
    #    aliases[k].append('Regno di Spagna')
    if k == 'Madonna':
        aliases[k].append('Madonna (entertainer)')
        aliases[k].append('Madonna (cantante)')
    if k == "corso d'acqua":
        aliases[k].append('Body of water')
        aliases[k].append("Corso d'acqua")
    #if name == 'Mar Mediterraneo':
    #if k == 'Germania':
    #    aliases[k].append('Repubblica Federale di Germania')
missing = list()

if args.model == 'wikipedia2vec':

    if args.language == 'en':
        model_path = '/import/cogsci/andrea/dataset/word_vectors/wikipedia2vec/enwiki_20180420_win10_500d.pkl'
        desired_shape = 500
    else:
        model_path = '/import/cogsci/andrea/dataset/word_vectors/wikipedia2vec/itwiki_20180420_300d.pkl'
        desired_shape = 300

    wikipedia2vec = Wikipedia2Vec.load(model_path)

    vectors = dict()
    for name, als in aliases.items():
        ent_vecs = list()
        for al in als:
            try:
                if name not in ['politico', 'scrittore', 'musicista', 'attore', 'stato', 'città', 'monumento']:
                    w_vec = wikipedia2vec.get_entity_vector(al)
                else:
                    w_vec = wikipedia2vec.get_word_vector(al)
                ent_vecs.append(w_vec)
            except KeyError:
                missing.append(al)
        vectors[name] = numpy.average(ent_vecs, axis=0)
    for k, v in vectors.items():
        #print(k)
        assert v.shape == (desired_shape,)

if args.model == 'transe':

    ernie_path = os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', \
                               'word_vectors', 'ERNIE')

    ent_2_q = entity_to_Q_id(ernie_path)
    q_2_line = Q_to_line(ernie_path)

    print('Now loading all the TransE vectors...')
    all_vectors = get_transe_vectors(ernie_path)
    assert len(all_vectors)-1 == max([v for k, v in q_2_line.items()])

    vectors = dict()      

    print('Now obtaining all the relevant TransE vectors...')
    for name, als in aliases.items():
        ent_vecs = list()
        for al in als:
            try:
                if name not in ['politico', 'scrittore', 'musicista', "corso d'acqua", 'attore', 'stato', 'città', 'monumento']:
                    entity_id = ent_2_q[al]
                else:
                    entity_id = ent_2_q[al.capitalize()]
            except KeyError:
                missing.append(al)
                continue
            line = q_2_line[entity_id]
            vector = numpy.array(all_vectors[line], dtype=numpy.single)
            assert vector.shape == (100,)
            ent_vecs.append(vector)
        vectors[name] = numpy.average(ent_vecs, axis=0)
    for k, v in vectors.items():
        assert v.shape == (100,)
if args.model == 'w2v':
    if args.language == 'it':
        path = os.path.join('/','import', 'cogsci', 'andrea',
                            'dataset', 'word_vectors',
                            'word2vec_wexea_entities_it_win_10_min_5_neg_1e-5_size_300',
                            'word2vec_wexea_entities_it_win_10_min_5_neg_1e-5_size_300.model')
    if args.language == 'en':
        path = os.path.join('/','import', 'cogsci', 'andrea',
                            'dataset', 'word_vectors',
                            'word2vec_wexea_entities_en_win_10_min_100_neg_1e-5_size_300',
                            'word2vec_wexea_entities_en_win_10_min_100_neg_1e-5_size_300.model')
    w2v = Word2Vec.load(path)
    desired_shape = 300
    missing = list()
    used = list()

    vectors = dict()
    for name, als in aliases.items():


        ent_vecs = list()
        for al in als:
            al = re.sub('[^A-za-z0-9]', '_', al).lower().strip()
            try:
                w_vec = w2v.wv.get_vector(al)
                ent_vecs.append(w_vec)
                used.append(al)
            except KeyError:
                missing.append(al)
        assert len(ent_vecs) >= 1
        vectors[name] = numpy.average(ent_vecs, axis=0)
    for k, v in vectors.items():
        #print(k)
        assert v.shape == (desired_shape,)

if 'sentence' in args.model:

    args.corpus = 'wikipedia'
    args.corpus_portion = 'entity_sentences'
    args.layer = 'top_four'
    args.individuals = True
    args.random = False
    vecs_file, rankings_folder = load_vec_files(args, args.model)
    if args.model == 'w2v_sentence':
        if args.language == 'it':
            path = os.path.join('/','import', 'cogsci', 'andrea',
                                'dataset', 'word_vectors',
                                'word2vec_wexea_entities_it_win_10_min_5_neg_1e-5_size_300',
                                'word2vec_wexea_entities_it_win_10_min_5_neg_1e-5_size_300.model')
            spacy_model = spacy.load("it_core_news_lg")
        if args.language == 'en':
            path = os.path.join('/','import', 'cogsci', 'andrea',
                                'dataset', 'word_vectors',
                                'word2vec_wexea_entities_en_win_10_min_100_neg_1e-5_size_300',
                                'word2vec_wexea_entities_en_win_10_min_100_neg_1e-5_size_300.model')
            spacy_model = spacy.load("en_core_web_lg")
        w2v = Word2Vec.load(path)
        desired_shape = 300
    if args.model == 'wikipedia2vec_sentence':
        if args.language == 'en':
            model_path = '/import/cogsci/andrea/dataset/word_vectors/wikipedia2vec/enwiki_20180420_win10_500d.pkl'
            desired_shape = 500
            spacy_model = spacy.load("en_core_web_lg")
        else:
            model_path = '/import/cogsci/andrea/dataset/word_vectors/wikipedia2vec/itwiki_20180420_300d.pkl'
            desired_shape = 300
            spacy_model = spacy.load("it_core_news_lg")
        wikipedia2vec = Wikipedia2Vec.load(model_path)

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
                if args.model == 'w2v_sentence':
                    replacements = [re.sub('[^A-Za-z0-9]', '_', match) for match in matches]
                    for original, repl in zip(matches, replacements):
                        line = line.replace('[[{}]]'.format(original), repl)
                    #line = re.sub('[^A-Za-z0-9_ ]', ' ', line).lower().split()
                    line = re.sub('[^A-Za-z0-9_ ]', ' ', line)
                    line = line.lower()
                elif args.model == 'wikipedia2vec_sentence':
                    replacements = [match for match in matches]
                    for original, repl in zip(matches, replacements):
                        line = line.replace('[[{}]]'.format(original), repl)
                if len(line.split()) > 10:
                    spacy_line = spacy_model(line)
                    line_vec = list()
                    #print(line)
                    ### entities
                    if args.model == 'wikipedia2vec_sentence':
                        for ent in spacy_line.ents:
                            try:
                                w_vec = wikipedia2vec.get_entity_vector(ent.text)
                                #print(ent.text)
                                line_vec.append(w_vec)
                                used.append(ent.text)
                            except KeyError:
                                missing.append(ent.text)
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
                            if args.model == 'w2v_sentence':
                                w_vec = w2v.wv.get_vector(tok.text)
                            if args.model == 'wikipedia2vec_sentence':
                                w_vec = wikipedia2vec.get_word_vector(tok.text.lower())
                                #if tok.ent_iob_ == 'O':
                                #    w_vec = wikipedia2vec.get_word_vector(tok.text.lower())
                                #else:
                                #    #print(tok.text)
                                #    continue
                            line_vec.append(w_vec)
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
    for k, v in vectors.items():
        #print(k)
        for line, vec in v:
            assert vec.shape == (desired_shape,)
    ### writing to file the vectors

    with open(vecs_file, 'w') as o:
        for stim, vecz in vectors.items():
            for line, vec in vecz:
                line = line.replace('\t', ' ')
                o.write('{}\t{}\t'.format(stim, line))
                for dim in vec:
                    o.write('{}\t'.format(float(dim)))
                o.write('\n')

else:

    with open(os.path.join(out_folder, 'exp_{}_{}_{}_vectors.tsv'.format(args.experiment_id, args.model, args.language)), 'w') as o:
        o.write('entity\t{}_entity_vector\n'.format(args.model))
        for k, vec in vectors.items():
            o.write('{}\t'.format(k))
            for dim in vec:
                o.write('{}\t'.format(dim))
            o.write('\n')
