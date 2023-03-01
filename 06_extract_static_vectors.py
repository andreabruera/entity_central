import argparse
import numpy
import os

from qwikidata.linked_data_interface import get_entity_dict_from_api
from tqdm import tqdm
from wikipedia2vec import Wikipedia2Vec

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
                                ],
                    required=True, help='Which model?')
args = parser.parse_args()

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
    main_alias = ent_dict['labels']['en']['value']
    aliases[k] = [main_alias]
    if 'en' in ent_dict['aliases'].keys():
        for al in ent_dict['aliases']['en']:
            if len(al) > 2:
                aliases[k].append(al['value'])
    if k == 'Madonna':
        aliases[k].append('Madonna (entertainer)')
    if k == "corso d'acqua":
        aliases[k].append('Body of water')
missing = list()

if args.model == 'wikipedia2vec':

    if args.language == 'en':
        model_path = '/import/cogsci/andrea/dataset/word_vectors/wikipedia2vec/enwiki_20180420_win10_500d.pkl'
    else:
        model_path = '/import/cogsci/andrea/dataset/word_vectors/wikipedia2vec/itwiki_20180420_300d.pkl'

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
        assert v.shape == (500,)

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

out_folder = 'static_vectors'
os.makedirs(out_folder, exist_ok=True)

with open(os.path.join(out_folder, 'exp_{}_{}_{}_vectors.tsv'.format(args.experiment_id, args.model, args.language)), 'w') as o:
    o.write('entity\t{}_entity_vector\n'.format(args.model))
    for k, vec in vectors.items():
        o.write('{}\t'.format(k))
        for dim in vec:
            o.write('{}\t'.format(dim))
        o.write('\n')
