import numpy
import os
import re

def read_sentences_folder(args):

    out_folder = os.path.join('sentences', args.corpus, args.language)
    os.makedirs(out_folder, exist_ok=True)

    return out_folder

def return_entity_file(entity):
    entity = '{}{}'.format(entity[0].capitalize(), entity[1:])
    wiki_file = '{}.txt'.format(re.sub('[^a-zA-Z0-9]', '_', entity))
    return wiki_file

def load_comp_model_name(args):
    if args.model == 'ITGPT2medium':
        model_name = 'GroNLP/gpt2-medium-italian-embeddings'
        computational_model = 'gpt2'
        out_shape = (1024, )
    if args.model == 'MBERT':
        model_name = 'bert-base-multilingual-cased'
        computational_model = 'MBERT'
        out_shape = (768, )
    if args.model == 'xlm-roberta-large':
        model_name = 'xlm-roberta-large'
        computational_model = 'xlm-roberta-large'
        out_shape = (1024, )

    return model_name, computational_model, out_shape

def read_full_wiki_vectors(vecs_file, out_shape):
    with open(vecs_file, 'r') as o:
        lines = [l.strip().split('\t') for l in o.readlines()]
    entity_vectors = dict()
    all_sentences = dict()
    for l in lines:
        vec = numpy.array(l[2:], dtype=numpy.float64)
        assert vec.shape == out_shape
        try:
            entity_vectors[l[0]].append(vec)
            all_sentences[l[0]].append(l[1])
        except KeyError:
            entity_vectors[l[0]] = [vec]
            all_sentences[l[0]] = [l[1]]

    return entity_vectors, all_sentences

def load_vec_files(args, computational_model):

    base_folder = os.path.join(
                              'contextualized_vector_selection', 
                              args.corpus,
                              computational_model,
                              args.layer,
                              )
    vecs_folder = os.path.join(
                              base_folder,
                              'phrase_vectors',
                              )

    os.makedirs(vecs_folder, exist_ok=True)
    vecs_file= os.path.join(
                            vecs_folder, 
                            'all.vectors'.format(computational_model)
                            )
    rankings_folder = os.path.join(
                              base_folder,
                              'rankings',
                              )

    os.makedirs(rankings_folder, exist_ok=True)

    return vecs_file, rankings_folder
