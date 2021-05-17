import os
import collections
import numpy

from tqdm import tqdm

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

def transe(entities_and_sentences_dict, args):

    ernie_path = os.path.join('/', 'import', 'cogsci', 'andrea', 'dataset', \
                               'word_vectors', 'ERNIE')

    ent_2_q = entity_to_Q_id(ernie_path)
    q_2_line = Q_to_line(ernie_path)

    print('Now loading all the TransE vectors...')
    all_vectors = get_transe_vectors(ernie_path)
    assert len(all_vectors)-1 == max([v for k, v in q_2_line.items()])

    transe_vectors = dict()      

    print('Now obtaining all the relevant TransE vectors...')
    for entity, sentences in tqdm(entities_and_sentences_dict.items()):        
        if entity == 'Sagrada_Família':
            entity = 'Sagrada Família'

        try:
            entity_id = ent_2_q[entity]
        except KeyError:
            print('missing: {}'.format(entity))
        
        line = q_2_line[entity_id]
        vector = numpy.array(all_vectors[line], dtype=numpy.single)
        assert vector.shape == (100,)
        
        transe_vectors[entity] = [('TransE dimensions for {}'.format(entity), \
                                   [vector])]

    return transe_vectors
